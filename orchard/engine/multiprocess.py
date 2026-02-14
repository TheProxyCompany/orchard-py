import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from sys import platform

import pynng

from orchard.ipc import endpoints as ipc_endpoints

logger = logging.getLogger(__name__)


def _pid_is_zombie(pid: int) -> bool:
    if platform != "darwin":
        return False
    result = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip().startswith("Z")


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _pid_is_zombie(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def read_pid_file(pid_file: Path) -> int | None:
    try:
        content = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        value = int(content)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def write_pid_file(pid_file: Path, pid: int) -> None:
    pid_file.write_text(f"{pid}\n", encoding="utf-8")


def read_ref_pids(ref_file: Path) -> list[int]:
    try:
        if not ref_file.exists():
            return []
        content = ref_file.read_text(encoding="utf-8")
        data = json.loads(content) if content else []
    except (OSError, json.JSONDecodeError):
        return []

    pids: list[int] = []
    for entry in data:
        try:
            pid = int(entry)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            pids.append(pid)
    return pids


def write_ref_pids(ref_file: Path, pids: Iterable[int]) -> None:
    unique = []
    seen: set[int] = set()
    for pid in pids:
        if pid <= 0 or pid in seen:
            continue
        unique.append(pid)
        seen.add(pid)

    if unique:
        tmp = ref_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(unique), encoding="utf-8")
        tmp.replace(ref_file)
    else:
        ref_file.unlink(missing_ok=True)


def filter_alive_pids(pids: Iterable[int]) -> list[int]:
    alive: list[int] = []
    seen: set[int] = set()
    for pid in pids:
        if pid in seen:
            continue
        if pid_is_alive(pid):
            alive.append(pid)
            seen.add(pid)
    return alive


# ENGINE PROCESS MANAGEMENT


def stop_engine_process(pid: int, timeout: float = 15.0) -> bool:
    """Sends signals and waits for a process to exit. Returns True on graceful exit."""
    try:
        os.kill(pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError) as e:
        logger.debug("Failed to send SIGINT to process %d: %s", pid, e)
        return not pid_is_alive(pid)

    try:
        if wait_for_exit(pid, timeout=timeout):
            return True
    except Exception:
        logger.warning(
            "Interrupt received while waiting for engine %d to stop; letting shutdown continue in the background.",
            pid,
        )
        spawn_async_reaper(pid)
        raise

    logger.warning("Engine did not respond to SIGINT. Sending SIGTERM.")
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return not pid_is_alive(pid)

    try:
        if wait_for_exit(pid, timeout=timeout):
            return True
    except KeyboardInterrupt:
        logger.warning(
            "Interrupt received while waiting for engine %d to respond to SIGTERM; allowing background cleanup.",
            pid,
        )
        spawn_async_reaper(pid)
        raise

    logger.error("Engine did not respond to SIGTERM. Sending SIGKILL.")
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    return not pid_is_alive(pid)


def wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    waitpid_supported = True
    while time.monotonic() < deadline:
        if waitpid_supported:
            try:
                finished_pid, _ = os.waitpid(pid, os.WNOHANG)
                if finished_pid == pid:
                    return True
            except ChildProcessError:
                waitpid_supported = False
        if not pid_is_alive(pid):
            return True
        time.sleep(0.05)
    if waitpid_supported:
        try:
            finished_pid, _ = os.waitpid(pid, os.WNOHANG)
            if finished_pid == pid:
                return True
        except ChildProcessError:
            waitpid_supported = False
    return not pid_is_alive(pid)


def reap_engine_process(pid: int) -> None:
    """Waits on the child process to clear the zombie entry."""
    try:
        while True:
            try:
                os.waitpid(pid, 0)
                break
            except InterruptedError:
                continue
    except ChildProcessError:
        # Not a child (already reaped elsewhere); nothing to do.
        return


def spawn_async_reaper(pid: int) -> None:
    thread = threading.Thread(
        target=reap_engine_process,
        args=(pid,),
        name=f"orchard-reaper-{pid}",
        daemon=True,
    )
    thread.start()


def wait_for_engine_ready(
    pid_file: Path,
    startup_timeout: float,
    process_alive_check: Callable[[], bool] | None = None,
) -> int:
    logger.info("Waiting for telemetry heartbeat from engine...")
    temp_sub_socket: pynng.Sub0 | None = None
    try:
        temp_sub_socket = pynng.Sub0()
        telemetry_topic = ipc_endpoints.EVENT_TOPIC_PREFIX + b"telemetry"
        temp_sub_socket.subscribe(telemetry_topic)
        temp_sub_socket.dial(ipc_endpoints.RESPONSE_URL, block=False)
        temp_sub_socket.recv_timeout = 250

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process_alive_check and not process_alive_check():
                raise RuntimeError(
                    "orchard engine exited before signaling readiness; check the engine log for details."
                )
            try:
                msg = temp_sub_socket.recv_msg()
            except KeyboardInterrupt:
                logger.warning(
                    "Keyboard interrupt received while waiting for telemetry heartbeat"
                )
                break
            except pynng.Timeout:
                continue
            if msg is None:
                continue
            parts = msg.bytes.split(b"\x00", 1)
            if len(parts) < 2:
                logger.warning(
                    "Discarding malformed event message while waiting for telemetry."
                )
                continue
            topic_part, json_body = parts
            if topic_part != telemetry_topic:
                logger.debug(
                    "Ignoring unexpected startup topic '%s'",
                    topic_part.decode("utf-8", errors="ignore"),
                )
                continue

            try:
                payload = json.loads(json_body) if json_body else {}
            except json.JSONDecodeError as exc:
                logger.warning("Discarding malformed telemetry payload: %s", exc)
                continue

            health = payload.get("health") if isinstance(payload, dict) else None
            engine_pid = health.get("pid") if isinstance(health, dict) else None

            if not isinstance(engine_pid, int) or engine_pid <= 0:
                logger.warning(
                    "Telemetry payload missing valid PID; waiting for next heartbeat."
                )
                continue

            write_pid_file(pid_file, engine_pid)
            logger.info(
                "Received telemetry heartbeat. Engine PID %d recorded.",
                engine_pid,
            )

            return engine_pid
        else:
            raise TimeoutError(
                f"Timed out after {startup_timeout}s waiting for telemetry heartbeat from engine."
            )
    except pynng.Timeout as e:
        raise TimeoutError(
            f"Timed out after {startup_timeout}s waiting for telemetry heartbeat from engine."
        ) from e
    finally:
        if temp_sub_socket:
            temp_sub_socket.close()

    return -1

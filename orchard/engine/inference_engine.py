from __future__ import annotations

import asyncio
import atexit
import logging
import os
import random
import subprocess
import threading
from pathlib import Path

import dotenv
from filelock import FileLock

from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import ModelRegistry
from orchard.clients import Client, get_client
from orchard.engine.fetch import (
    check_for_updates_async,
    get_available_update,
    get_engine_path,
)
from orchard.engine.global_context import GlobalContext, global_context
from orchard.engine.io import close_sockets, get_engine_file_paths, initialize_sockets
from orchard.engine.multiprocess import (
    filter_alive_pids,
    pid_is_alive,
    read_pid_file,
    read_ref_pids,
    reap_engine_process,
    stop_engine_process,
    wait_for_engine_ready,
    write_ref_pids,
)

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

_log_handler: logging.Handler | None = None
_atexit_registered = False
_dispatcher_cleanup_registered = False


_LOCK_TIMEOUT_S = 30.0
_DEFAULT_ENGINE_PORT = 8000


class InferenceEngine:
    """Process-safe manager that launches and reference-counts orchard engine."""

    def __init__(
        self,
        client_log_file: Path | None = None,
        engine_log_file: Path | None = None,
        startup_timeout: float = 60.0,
        load_models: list[str] | None = None,
    ):
        check_for_updates_async()  # Fire-and-forget background update check

        self._paths = get_engine_file_paths(client_log_file, engine_log_file)
        self._setup_logging(client_log_file, engine_log_file)
        self._startup_timeout = float(startup_timeout)
        self._lock = FileLock(str(self._paths.lock_file), timeout=_LOCK_TIMEOUT_S)
        self._engine_bin = get_engine_path()
        self._lease_active = False
        self._closed = False
        self._launch_process: subprocess.Popen | None = None

        global _atexit_registered
        if not _atexit_registered:
            atexit.register(InferenceEngine.shutdown)
            _atexit_registered = True

        self._acquire_lease_and_init_global_context()
        if load_models:
            asyncio.run(self.load_models(load_models))

    def __enter__(self) -> InferenceEngine:
        if self._closed:
            raise RuntimeError("InferenceEngine instance already closed.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> InferenceEngine:
        # The startup logic is synchronous, so this is straightforward
        if self._closed:
            raise RuntimeError("InferenceEngine instance already closed.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # The close logic is synchronous and blocking, so we run it in an executor
        # to avoid blocking the asyncio event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close)

    async def load_models(self, model_ids: list[str]):
        logger.info("Loading models: %s", ", ".join(model_ids))
        tasks = [
            asyncio.create_task(self.load_model(model_id)) for model_id in model_ids
        ]
        await asyncio.gather(*tasks)
        logger.info("Models loaded: %s", ", ".join(model_ids))

    async def load_model(self, model_id: str):
        """
        Requests the engine to load a model and waits for it to become ready.
        """
        if not global_context.model_registry:
            raise RuntimeError("Model registry is not initialized.")
        await global_context.model_registry.ensure_loaded(model_id)

    def client(self, model_id: str | None = None) -> Client:
        if self._closed:
            raise RuntimeError("InferenceEngine has been closed.")
        if global_context.ipc_state is None:
            raise RuntimeError("IPC state is not initialized.")
        if global_context.model_registry is None:
            raise RuntimeError("Model registry is not initialized.")

        return get_client(
            model_id,
            ipc_state=global_context.ipc_state,
            model_registry=global_context.model_registry,
        )

    def ipc_state(self) -> IPCState:
        if self._closed:
            raise RuntimeError("InferenceEngine has been closed.")
        if global_context.ipc_state is None:
            raise RuntimeError("IPC state is not initialized.")
        return global_context.ipc_state

    def model_registry(self) -> ModelRegistry:
        if self._closed:
            raise RuntimeError("InferenceEngine has been closed.")
        if global_context.model_registry is None:
            raise RuntimeError("Model registry is not initialized.")
        return global_context.model_registry

    def close(self) -> None:
        if self._closed:
            return

        release_process_lease = False
        if self._lease_active:
            with global_context.lock:
                if global_context.ref_count > 0:
                    global_context.ref_count -= 1
                release_process_lease = global_context.ref_count == 0

        try:
            if not self._lease_active or not release_process_lease:
                return

            with self._lock:
                refs = read_ref_pids(self._paths.refs_file)
                alive_refs = filter_alive_pids(refs)
                current_pid = os.getpid()
                alive_refs = [pid for pid in alive_refs if pid != current_pid]

                engine_pid = read_pid_file(self._paths.pid_file)
                engine_running = engine_pid is not None and pid_is_alive(engine_pid)

                self.shutdown_global_context(
                    global_context,
                    decrement_ref=False,
                )

                if not alive_refs:
                    if engine_running and engine_pid is not None:
                        self._stop_engine_locked(engine_pid)
                    else:
                        self._paths.pid_file.unlink(missing_ok=True)
                        self._paths.ready_file.unlink(missing_ok=True)
                    write_ref_pids(self._paths.refs_file, [])
                else:
                    write_ref_pids(self._paths.refs_file, alive_refs)
        finally:
            self._lease_active = False
            self._closed = True

    def _launch_engine_locked(self) -> None:
        cache_dir = self._paths.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        ready_file = self._paths.ready_file
        ready_file.unlink(missing_ok=True)
        self._paths.pid_file.unlink(missing_ok=True)

        cmd = [str(self._engine_bin)]
        process = None
        try:
            with open(self.engine_log_path, "w") as log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=log_handle,
                    start_new_session=True,
                )
                self._launch_process = process
        except Exception as e:
            if process:
                process.terminate()
                process.wait(timeout=10)
            raise RuntimeError(
                f"Failed to launch orchard engine executable: {e}"
            ) from e

    def _acquire_lease_and_init_global_context(self):
        """
        Main startup logic. Manages engine process and in-process app state.
        This entire method is protected by a system-wide file lock.
        """
        if self._closed or self._lease_active:
            return

        with self._lock:
            refs = read_ref_pids(self._paths.refs_file)
            alive_refs = filter_alive_pids(refs)

            engine_pid = read_pid_file(self._paths.pid_file)
            engine_running = engine_pid is not None and pid_is_alive(engine_pid)
            if not engine_running:
                engine_pid = None
                self._paths.pid_file.unlink(missing_ok=True)
                self._paths.ready_file.unlink(missing_ok=True)

            if not engine_running and not alive_refs:
                logger.debug(
                    "Inference engine not running. Launching new instance as leader."
                )
                try:
                    self._launch_engine_locked()
                    engine_pid = self._wait_for_engine_ready()
                except Exception:
                    self._cleanup_failed_launch()
                    raise

            current_pid = os.getpid()
            if current_pid not in alive_refs:
                alive_refs.append(current_pid)

            write_ref_pids(self._paths.refs_file, alive_refs)

        try:
            self.initialize_global_context(global_context)
        except Exception:
            # Roll back PID registration if initialization fails
            with self._lock:
                refs = read_ref_pids(self._paths.refs_file)
                alive_refs = [
                    pid for pid in filter_alive_pids(refs) if pid != os.getpid()
                ]
                write_ref_pids(self._paths.refs_file, alive_refs)
            raise

        self._lease_active = True

    def _wait_for_engine_ready(self) -> int:
        logger.info("Waiting for ENGINE_READY event from the C++ engine...")
        return wait_for_engine_ready(
            self._paths.pid_file,
            self._startup_timeout,
            process_alive_check=lambda: self._launch_process is not None
            and self._launch_process.poll() is None,
        )

    def _cleanup_failed_launch(self) -> None:
        if self._launch_process and self._launch_process.poll() is None:
            self._launch_process.terminate()
            try:
                self._launch_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._launch_process.kill()
                self._launch_process.wait(timeout=5)
        self._launch_process = None
        self._paths.pid_file.unlink(missing_ok=True)
        self._paths.ready_file.unlink(missing_ok=True)

    def _stop_engine_locked(self, pid: int) -> None:
        if not pid_is_alive(pid):
            logger.debug("orchard engine PID %s already exited.", pid)
            self._paths.pid_file.unlink(missing_ok=True)
            self._paths.ready_file.unlink(missing_ok=True)
            return

        if not stop_engine_process(pid, timeout=5.0):
            logger.warning("Failed to stop orchard engine PID %s.", pid)
            raise RuntimeError(f"Failed to stop orchard engine PID {pid}.")

        reap_engine_process(pid)
        self._paths.pid_file.unlink(missing_ok=True)
        self._paths.ready_file.unlink(missing_ok=True)
        logger.info("orchard engine PID %s stopped and readiness cleared.", pid)

    def _setup_logging(
        self, client_log_file: Path | None = None, engine_log_file: Path | None = None
    ) -> None:
        self.engine_log_path = (
            Path(engine_log_file) if engine_log_file else self._paths.engine_log_file
        )
        self.engine_log_path.parent.mkdir(parents=True, exist_ok=True)

        global _log_handler
        if _log_handler is not None:
            return

        resolved_client_log = (
            client_log_file if client_log_file else self._paths.client_log_file
        )
        resolved_client_log.parent.mkdir(parents=True, exist_ok=True)
        log_level = os.getenv("LOG_LEVEL", "INFO")
        handler = logging.FileHandler(resolved_client_log, mode="w", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
        _log_handler = handler
        logger.info(
            "InferenceEngine initialized with client log file: %s",
            resolved_client_log,
        )

    @staticmethod
    def shutdown(timeout: float = 15.0) -> bool:
        """
        Forcefully stops the shared orchard engine process, bypassing reference counts.

        This is an administrative action that will terminate the engine even if
        other clients are connected.

        Returns:
            True if the engine was stopped gracefully, False otherwise.
        """
        if update := get_available_update():
            logger.info("Orchard upgrade available: %s", update)
            print(
                f"Orchard upgrade available. Run `orchard upgrade` to install: {update}"
            )

        paths = get_engine_file_paths(None, None)
        lock = FileLock(str(paths.lock_file), timeout=_LOCK_TIMEOUT_S)

        with lock:
            pid = read_pid_file(paths.pid_file)
            if pid is None or not pid_is_alive(pid):
                logger.info("Engine is not running. Cleaning up any stale files.")
                paths.pid_file.unlink(missing_ok=True)
                paths.ready_file.unlink(missing_ok=True)
                paths.refs_file.unlink(missing_ok=True)
                return True

            logger.info("Sending shutdown signal (SIGINT) to engine process %d.", pid)

            # Use a private helper to contain the actual signaling logic
            shutdown_successful = stop_engine_process(pid, timeout)

            if shutdown_successful:
                paths.pid_file.unlink(missing_ok=True)
                paths.ready_file.unlink(missing_ok=True)
                paths.refs_file.unlink(missing_ok=True)
                reap_engine_process(pid)
                logger.info("Engine process %d terminated gracefully.", pid)
                return True

            logger.critical(
                "Engine process %d failed to stop. State files preserved for manual intervention.",
                pid,
            )
            raise RuntimeError(
                f"Force shutdown failed for engine process {pid}. Manual cleanup required."
            )

    @staticmethod
    def generate_response_channel_id() -> int:
        pid_component = os.getpid() & 0xFFFFFFFF
        random_component = random.getrandbits(32)
        channel_id = (pid_component << 32) | random_component
        if channel_id == 0:
            channel_id = 1
        return channel_id

    @staticmethod
    def initialize_global_context(ctx: GlobalContext) -> None:
        """Initializes the singleton app state components in a thread-safe way."""
        with ctx.lock:
            ctx.ref_count += 1
            if ctx.initialized:
                return

            logger.info("Initializing global Python application state for PIE.")
            response_channel_id = InferenceEngine.generate_response_channel_id()

            try:
                # Initialize IPC State container
                ctx.ipc_state = IPCState(ctx)
                ctx.ipc_state.response_channel_id = response_channel_id

                # Create Model Registry
                ctx.model_registry = ModelRegistry(ctx.ipc_state)

                # Connect sockets (Engine is guaranteed ready by this point)
                initialize_sockets(ctx.ipc_state, response_channel_id)

                # Start the dispatcher thread
                def _dispatcher_thread_target(ipc_state: IPCState):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(IPCState.run_ipc_listener(ipc_state))
                    finally:
                        loop.close()

                ctx.dispatcher_thread = threading.Thread(
                    target=_dispatcher_thread_target,
                    args=(ctx.ipc_state,),
                    name="pie-async-dispatcher",
                    daemon=True,
                )
                ctx.dispatcher_thread.start()

                # Ensure sockets close and dispatcher joins on abrupt process exit (e.g., Ctrl-C)
                def _cleanup_dispatcher():
                    close_sockets(ctx.ipc_state)
                    if ctx.dispatcher_thread and ctx.dispatcher_thread.is_alive():
                        ctx.dispatcher_thread.join(timeout=5.0)

                global _dispatcher_cleanup_registered
                if not _dispatcher_cleanup_registered:
                    atexit.register(_cleanup_dispatcher)
                    _dispatcher_cleanup_registered = True

                ctx.initialized = True
                logger.info("Global Python application state initialized.")

            except Exception:
                ctx.ref_count -= 1
                logger.exception("Failed to initialize Python application state.")

                if ctx.dispatcher_thread and ctx.dispatcher_thread.is_alive():
                    ctx.dispatcher_thread.join(timeout=5.0)

                # Clean up partial state
                close_sockets(ctx.ipc_state)
                ctx.dispatcher_thread = None
                ctx.ipc_state = None
                ctx.model_registry = None
                raise

    @staticmethod
    def shutdown_global_context(ctx: GlobalContext, decrement_ref: bool = True) -> None:
        with ctx.lock:
            if decrement_ref and ctx.ref_count > 0:
                ctx.ref_count -= 1

            # Only shut down if ref_count hits zero
            if not ctx.initialized or ctx.ref_count > 0:
                return

            logger.info("Shutting down global Python application state.")

            # Close sockets to break the dispatcher loop
            close_sockets(ctx.ipc_state)

            if ctx.dispatcher_thread and ctx.dispatcher_thread.is_alive():
                ctx.dispatcher_thread.join(timeout=5.0)

            ctx.dispatcher_thread = None
            ctx.ipc_state = None
            ctx.model_registry = None
            ctx.initialized = False

            logger.info("Global Python application state shut down.")

            global _log_handler

            if _log_handler is not None:
                root_logger = logging.getLogger()
                root_logger.removeHandler(_log_handler)
                _log_handler.close()
                _log_handler = None


__all__ = ["InferenceEngine"]

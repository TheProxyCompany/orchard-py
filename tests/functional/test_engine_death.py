"""Real PIE transport and server teardown, without loading a model."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_mode", ["death", "close"])
def test_engine_death_settles_streams_and_server_cleanup(tmp_path, exit_mode):
    assert _run_engine_process(tmp_path, exit_mode) == {
        "streams_settled": 36,
        "management_settled": 3,
        "model_calls_settled": 2,
    }


def test_concurrent_close_preserves_the_new_instance_lease(tmp_path):
    assert _run_engine_process(tmp_path, "handoff") == {
        "new_lease_preserved": True,
        "final_lease_released": True,
    }


def _run_engine_process(tmp_path, exit_mode):
    # A separate process keeps this death test out of the session engine used
    # by the model tests and gives both Python and PIE the same private IPC root.
    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        ORCHARD_CACHE_ROOT=str(tmp_path / "cache"),
        ORCHARD_IPC_ROOT=str(tmp_path / "ipc"),
        ORCHARD_TEST_LOG_DIR=str(tmp_path / "logs"),
        PYTHONPATH=os.pathsep.join([str(project), str(project / "tests")]),
    )
    process = subprocess.Popen(
        [sys.executable, __file__, exit_mode],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            pid_file = tmp_path / "cache" / "engine.pid"
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                identity = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "ppid=,comm="],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3,
                ).stdout.split(None, 1)
                if (
                    len(identity) == 2
                    and int(identity[0]) == process.pid
                    and Path(identity[1].strip()).name == "proxy_inference_engine"
                ):
                    os.kill(pid, signal.SIGKILL)
        finally:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        pytest.fail("Engine-death subprocess did not exit:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    return json.loads(stdout.splitlines()[-1])


def _exercise_lease_handoff():
    from filelock import FileLock

    from orchard.engine.inference_engine import InferenceEngine
    from orchard.engine.multiprocess import pid_is_alive, read_pid_file, wait_for_exit

    engine = InferenceEngine(load_models=[], startup_timeout=20)
    process = engine._launch_process
    assert process is not None, "The private namespace must launch its own engine"
    started = threading.Event()
    finished = threading.Event()
    close_errors = []
    replacement = None

    def close():
        started.set()
        try:
            engine.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            finished.set()

    closing = threading.Thread(target=close)
    try:
        with FileLock(str(engine._paths.lock_file)):
            closing.start()
            assert started.wait(2)
            # The close must wait for the real lifecycle lock. Release during
            # its retry interval so the new constructor can acquire it first.
            time.sleep(0.015)
            assert not finished.is_set()
        replacement = InferenceEngine(load_models=[], startup_timeout=20)
        constructed_before_close = not finished.is_set()
        closing.join(timeout=10)
        assert not closing.is_alive()
        assert not close_errors, close_errors
        assert constructed_before_close, "Constructor did not overtake the waiting close"
        client = replacement.client()
        try:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                assert client.cancel_request(999999)["status"] == "accepted"
                assert read_pid_file(engine._paths.pid_file) == process.pid
                assert pid_is_alive(process.pid)
                time.sleep(0.05)
        finally:
            client.close()
        replacement.close()
        assert wait_for_exit(process.pid, 10), "The final lease did not release PIE"
        print(json.dumps({"new_lease_preserved": True, "final_lease_released": True}))
    finally:
        closing.join(timeout=10)
        if replacement is not None:
            replacement.close()
        engine.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


async def _exercise_engine_death(exit_mode):
    from conftest import live_server

    from orchard.app.model_registry import ModelEntry, ModelInfo, ModelLoadState
    from orchard.app.model_resolver import ResolvedModel
    from orchard.engine.inference_engine import InferenceEngine
    from orchard.ipc.serialization import _build_request_payload
    from orchard.server.routes._common import managed_stream_session

    engine = InferenceEngine(load_models=[], startup_timeout=20)
    process = engine._launch_process
    assert process is not None, "The private namespace must launch its own engine"
    state = engine.ipc_state()
    client = engine.client()
    server = live_server.__wrapped__(engine)
    streams = []
    management = []
    model_calls = []
    activation = None
    closing = None
    sent = []

    async def observed(condition, timeout=2):
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.01)

    async def stream(request_id):
        queue = asyncio.Queue()
        try:
            async with managed_stream_session(
                state, request_id, queue, cancel_on_exit=True
            ):
                payload = _build_request_payload(
                    request_id=request_id,
                    response_channel_id=state.response_channel_id,
                    model_id="transport-death-probe",
                    model_path=str(engine._paths.cache_dir),
                    request_type="generation",
                    prompts=[{"prompt": "No model is loaded in this transport test."}],
                )
                await state.send_request(payload)
                sent.append(request_id)
                delta = await state.next_delta(queue)
                assert delta["is_final_delta"]
                return delta.get("error") or delta["content"]
        except RuntimeError as exc:
            return str(exc)

    try:
        await anext(server)
        # Preserve live transport replies and rejected management commands.
        assert (await client.acancel_request(999999))["status"] == "accepted"
        assert (await asyncio.to_thread(client.cancel_request, 999996))[
            "status"
        ] == "accepted"
        live_result = await asyncio.wait_for(
            stream(await state.get_next_request_id()), 5
        )
        assert "disconnected" not in live_result
        rejected = await state.send_management_command(
            {"type": "transport_death_unknown_command"}
        )
        assert rejected["status"] == "unsupported", rejected
        assert not state.engine_dead
        sent.clear()

        os.kill(process.pid, signal.SIGSTOP)
        # Caller cancellation and a live-peer reply timeout retain their meaning.
        cancelled = asyncio.create_task(client.acancel_request(999998))
        await observed(state.management_lock.locked)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        with pytest.raises(TimeoutError):
            await state.cancel_request(999997, timeout=0.05)
        assert not state.engine_dead

        registry = engine.model_registry()
        source = ResolvedModel(
            canonical_id="transport-death-inspect",
            model_path=engine._paths.cache_dir,
            source="local_source",
        )
        model_calls.append(
            asyncio.create_task(
                registry._inspect_model_source(source.canonical_id, source)
            )
        )
        await observed(state.management_lock.locked)
        info = ModelInfo(
            model_id="transport-death-load",
            model_path=str(engine._paths.cache_dir),
            formatter=object(),
        )
        activation = asyncio.get_running_loop().create_future()
        registry._entries[info.model_id] = ModelEntry(
            state=ModelLoadState.ACTIVATING,
            info=info,
            activation_future=activation,
            activation_loop=asyncio.get_running_loop(),
        )
        model_calls.append(
            asyncio.create_task(
                registry._send_load_model_command(
                    requested_id=info.model_id, canonical_id=info.model_id, info=info
                )
            )
        )

        for index in range(36):
            request_id = await state.get_next_request_id()
            if index % 2:
                future = asyncio.run_coroutine_threadsafe(
                    stream(request_id), client._sync_loop
                )
                streams.append(asyncio.wrap_future(future))
            else:
                streams.append(asyncio.create_task(stream(request_id)))
        management = [
            asyncio.create_task(state.cancel_request(1000000 + index, timeout=None))
            for index in range(3)
        ]
        # The stopped real peer accepts some frames into its socket buffers;
        # the remaining sends and management exchanges stay in flight.
        await observed(
            lambda: (
                len(state.active_request_queues) == 36
                and 0 < len(sent) < 36
                and state.management_lock.locked()
            )
        )
        assert len(state.active_request_queues) == 36
        assert 0 < len(sent) < 36
        assert not any(task.done() for task in management + model_calls)
        if exit_mode == "death":
            process.kill()
            process.wait(timeout=3)
        else:
            closing = asyncio.create_task(asyncio.to_thread(engine.close))
            _, pending_close = await asyncio.wait([closing], timeout=15)
            assert not pending_close, "Normal close did not drain IPC operations"
            closing.result()
            assert process.poll() is None, "The engine was still alive during close"

        done, pending = await asyncio.wait(
            streams + management + model_calls, timeout=10
        )
        assert not pending, f"Engine death left {len(pending)} IPC operations waiting"
        assert len(done) == 41
        assert all(task.result() == "Engine process disconnected." for task in streams)
        assert all(
            isinstance(task.exception(), RuntimeError)
            and str(task.exception()) == "Engine process disconnected."
            for task in management
        )
        assert all(
            isinstance(task.exception(), RuntimeError)
            and "Engine process disconnected." in str(task.exception())
            for task in model_calls
        )
        assert registry.get_status(info.model_id)[0] == ModelLoadState.FAILED
        assert isinstance(activation.exception(), RuntimeError)
        assert not state.active_request_queues
        assert state.wait_for_inflight_drain(0)
        # This is the existing live_server fixture, including real HTTP health,
        # uvicorn shutdown, shared-client lifespan, and fixture-generator close.
        await server.aclose()
        assert not any(
            thread.name == "orchard-test-fastapi" and thread.is_alive()
            for thread in threading.enumerate()
        )
        print(
            json.dumps(
                {
                    "streams_settled": 36,
                    "management_settled": 3,
                    "model_calls_settled": 2,
                }
            )
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if closing is not None:
            await closing
        await observed(lambda: state.engine_dead, timeout=8)
        for task in streams + management + model_calls:
            if not task.done():
                task.cancel()
        if streams or management or model_calls:
            await asyncio.gather(
                *streams, *management, *model_calls, return_exceptions=True
            )
        if activation is not None:
            if activation.done() and not activation.cancelled():
                activation.exception()
            else:
                activation.cancel()
        await server.aclose()
        client.close()
        engine.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "death"
    if mode == "handoff":
        _exercise_lease_handoff()
    else:
        asyncio.run(_exercise_engine_death(mode))

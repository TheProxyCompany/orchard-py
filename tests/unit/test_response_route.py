"""How response deltas get from the engine to a request's queue.

The engine here is a stand-in that answers the way PIE does: a request that
asks for "pull_v1" gets its deltas pushed to the client's own endpoint with no
send buffer, so a slow client makes the sender wait; any other request gets
them published, where a slow client silently loses the oldest ones.
"""

import asyncio
import json
import logging
import struct
import tempfile
import threading
import time
from pathlib import Path

import pynng
import pytest

from orchard.app import ipc_dispatch
from orchard.app.ipc_dispatch import IPCDispatcher, IPCState, QueueRegistration
from orchard.app.model_registry import ModelRegistry
from orchard.engine import io as engine_io
from orchard.engine.global_context import GlobalContext
from orchard.engine.inference_engine import InferenceEngine
from orchard.ipc import endpoints
from orchard.ipc.serialization import _build_request_payload

STALL_S = 1.5
DELTAS_PER_SECOND = 4000
CHANNEL = 0xA1B2C3
OTHER_CHANNEL = 0xA1B2C4


class StandInEngine:
    def __init__(self, *, knows_pull_route: bool = True) -> None:
        self.knows_pull_route = knows_pull_route
        self.requests = pynng.Pull0(listen=endpoints.REQUEST_URL, recv_timeout=5000)
        self.published = pynng.Pub0(listen=endpoints.RESPONSE_URL)
        self.published.send_buffer_size = 1024
        self.management = pynng.Rep0(listen=endpoints.MANAGEMENT_URL)
        self.routes: dict[int, pynng.Push0] = {}

    def receive_request(self) -> dict:
        frame = self.requests.recv()
        return json.loads(frame[4 : 4 + struct.unpack_from("<I", frame, 0)[0]])

    def answer(self, request: dict, count: int, *, paced: bool = False) -> None:
        """Sends `count` numbered deltas for the request, the last one final."""
        channel = request["response_channel_id"]
        if self.knows_pull_route and request.get("response_transport") == "pull_v1":
            if channel not in self.routes:
                # PIE waits without limit; a test must fail instead of hanging.
                route = pynng.Push0(send_timeout=10_000)
                route.send_buffer_size = 0
                route.dial(f"ipc://{endpoints.response_pull_path(channel)}", block=True)
                self.routes[channel] = route
            send = self.routes[channel].send
        else:
            send = self.published.send

        topic = f"resp:{channel:x}:".encode()
        start = time.monotonic()
        for number in range(count):
            delta = {
                "request_id": request["request_id"],
                "number": number,
                "content": "x" * 1000,
                "is_final_delta": number == count - 1,
            }
            send(topic + json.dumps(delta).encode())
            if paced:
                time.sleep(
                    max(0.0, start + number / DELTAS_PER_SECOND - time.monotonic())
                )

    def publish_event(self, name: str, payload: dict) -> None:
        self.published.send(
            endpoints.EVENT_TOPIC_PREFIX
            + name.encode()
            + b"\x00"
            + json.dumps(payload).encode()
        )

    def close(self) -> None:
        for socket in (
            self.requests,
            self.published,
            self.management,
            *self.routes.values(),
        ):
            socket.close()


class Client:
    """The engine client's IPC half, started the way InferenceEngine starts it.

    `launched_engine` is what InferenceEngine records when this process
    launched the engine itself, as opposed to attaching to a running one."""

    def __init__(self, channel_id: int, *, launched_engine: bool) -> None:
        self.ctx = GlobalContext()
        self.state = self.ctx.ipc_state = IPCState(self.ctx)
        self.ctx.model_registry = ModelRegistry(self.state)
        self.state.response_channel_id = channel_id
        self.state.launched_engine = launched_engine
        engine_io.initialize_sockets(self.state, channel_id)
        self.ctx.dispatcher_thread = threading.Thread(
            target=lambda: asyncio.run(IPCState.run_ipc_listener(self.state)),
            daemon=True,
        )
        self.ctx.dispatcher_thread.start()

    async def request(self, request_id: int) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.state.active_request_queues[request_id] = QueueRegistration(
            asyncio.get_running_loop(), queue
        )
        await self.state.send_request(
            _build_request_payload(
                request_id=request_id,
                model_id="test-model",
                model_path="/tmp/test-model",
                request_type="generation",
                response_channel_id=self.state.response_channel_id,
                lossless_responses=self.state.lossless_responses,
                prompts=[{"prompt": "hello"}],
            )
        )
        return queue

    def close(self) -> None:
        self.state.active_request_queues.clear()
        InferenceEngine._request_dispatcher_shutdown(self.ctx)
        InferenceEngine._join_dispatcher_thread(self.ctx)
        InferenceEngine._close_sockets_if_dispatcher_stopped(self.ctx)


@pytest.fixture
def ipc_root(monkeypatch):
    # pytest's tmp_path is too long for a unix socket path (103 bytes on macOS).
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory).resolve()
        monkeypatch.setattr(endpoints, "IPC_ROOT", root)
        for name, file in (
            ("REQUEST_URL", "pie_requests.ipc"),
            ("RESPONSE_URL", "pie_responses.ipc"),
            ("MANAGEMENT_URL", "pie_management.ipc"),
        ):
            monkeypatch.setattr(endpoints, name, f"ipc://{root / file}")
        monkeypatch.setattr(ipc_dispatch, "RESPONSE_RECV_TIMEOUT_MS", 50)
        yield root


# These fixtures are not called `engine` and `client`: conftest.py has fixtures
# by those names that start the real engine and load every model.
@pytest.fixture
def stand_in(ipc_root):
    engine = StandInEngine()
    yield engine
    engine.close()


@pytest.fixture
def ipc_client(stand_in):
    started = Client(CHANNEL, launched_engine=True)
    yield started
    started.close()


@pytest.fixture
def shared_engine_client(stand_in):
    started = Client(CHANNEL, launched_engine=False)
    yield started
    started.close()


async def numbers_received(queue: asyncio.Queue) -> list[int]:
    """Everything that arrives up to the final delta, or until nothing has
    arrived for five seconds."""
    numbers = []
    try:
        while True:
            delta = await asyncio.wait_for(queue.get(), timeout=5.0)
            numbers.append(delta["number"])
            if delta["is_final_delta"]:
                break
    except TimeoutError:
        pass
    return numbers


def hold_the_receiver_once(monkeypatch, *, after: int, seconds: float) -> None:
    """Blocks the dispatcher thread once, the way a busy process does."""
    dispatch = IPCDispatcher.dispatch
    seen = 0

    def stalled(self, ipc_state, msg_bytes):
        nonlocal seen
        seen += 1
        if seen == after:
            time.sleep(seconds)
        return dispatch(self, ipc_state, msg_bytes)

    monkeypatch.setattr(IPCDispatcher, "dispatch", stalled)


@pytest.mark.asyncio
async def test_deltas_reach_their_own_request_queue_in_order(stand_in, ipc_client):
    first = await ipc_client.request(1)
    second = await ipc_client.request(2)
    requests = [stand_in.receive_request(), stand_in.receive_request()]

    def interleave() -> None:
        for _ in range(100):
            for request in requests:
                stand_in.answer(request, 3)

    await asyncio.to_thread(interleave)

    # Both requests were answered over the client's one endpoint.
    assert list(stand_in.routes) == [CHANNEL]

    for queue in (first, second):
        numbers = [
            (await asyncio.wait_for(queue.get(), 5.0))["number"] for _ in range(300)
        ]
        assert numbers == [0, 1, 2] * 100
        assert queue.empty()


@pytest.mark.asyncio
async def test_a_receiver_that_stalls_loses_no_deltas(
    monkeypatch, stand_in, ipc_client
):
    """Over publish/subscribe, which is what a request got before it asked for
    the route, this same stall loses 2,776 of the 4,000 deltas and the stream
    still ends normally."""
    hold_the_receiver_once(monkeypatch, after=200, seconds=STALL_S)
    queue = await ipc_client.request(1)
    request = stand_in.receive_request()
    count = DELTAS_PER_SECOND  # one second of deltas, most of it inside the stall

    sender = asyncio.to_thread(stand_in.answer, request, count, paced=True)
    numbers, _ = await asyncio.gather(numbers_received(queue), sender)

    lost = count - len(numbers)
    assert lost == 0, f"{lost} of {count} deltas never arrived"
    assert numbers == list(range(count))


@pytest.mark.asyncio
async def test_an_engine_without_the_route_is_still_heard_and_reported(
    ipc_root, caplog
):
    engine = StandInEngine(knows_pull_route=False)
    client = Client(CHANNEL, launched_engine=True)
    try:
        queue = await client.request(1)
        with caplog.at_level(logging.WARNING, logger=ipc_dispatch.__name__):
            await asyncio.to_thread(engine.answer, engine.receive_request(), 50)
            assert await numbers_received(queue) == list(range(50))
    finally:
        client.close()
        engine.close()

    warnings = [r for r in caplog.records if "publish/subscribe" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_what_a_reaping_engine_publishes_is_delivered_without_the_warning(
    ipc_root, caplog
):
    # An engine that advertises lossless_responses publishes only what it
    # could not push to the endpoint, such as the error for a failed dial.
    engine = StandInEngine(knows_pull_route=False)
    client = Client(CHANNEL, launched_engine=False)
    client.state.engine_advertises_lossless = True
    try:
        queue = await client.request(1)
        with caplog.at_level(logging.WARNING, logger=ipc_dispatch.__name__):
            await asyncio.to_thread(engine.answer, engine.receive_request(), 5)
            assert await numbers_received(queue) == list(range(5))
    finally:
        client.close()
        engine.close()

    assert not [r for r in caplog.records if "publish/subscribe" in r.getMessage()]


@pytest.mark.asyncio
async def test_an_engine_this_process_launched_is_asked_for_the_route(
    stand_in, ipc_client
):
    # No capability was ever advertised: launching the engine is enough.
    queue = await ipc_client.request(1)
    request = stand_in.receive_request()

    assert request["response_transport"] == "pull_v1"
    await asyncio.to_thread(stand_in.answer, request, 5)
    assert await numbers_received(queue) == list(range(5))


@pytest.mark.asyncio
async def test_a_shared_engine_is_asked_for_the_route_once_it_advertises_it(
    stand_in, shared_engine_client
):
    client = shared_engine_client

    # The request on main, answered where main's requests are answered.
    queue = await client.request(1)
    request = stand_in.receive_request()
    assert "response_transport" not in request
    await asyncio.to_thread(stand_in.answer, request, 50)
    assert await numbers_received(queue) == list(range(50))
    assert stand_in.routes == {}

    # Publish/subscribe does not queue for a subscriber it has not seen yet.
    deadline = time.monotonic() + 5.0
    while not client.state.engine_advertises_lossless:
        assert time.monotonic() < deadline, "the model_loaded event never arrived"
        stand_in.publish_event(
            "model_loaded",
            {
                "model_id": "m",
                "capabilities": {"answer": [3], "lossless_responses": [1]},
            },
        )
        await asyncio.sleep(0.01)

    queue = await client.request(2)
    request = stand_in.receive_request()
    assert request["response_transport"] == "pull_v1"
    await asyncio.to_thread(stand_in.answer, request, 50)
    assert await numbers_received(queue) == list(range(50))
    assert list(stand_in.routes) == [CHANNEL]


@pytest.mark.asyncio
async def test_two_clients_with_the_same_request_ids_hear_only_their_own(
    stand_in, ipc_client
):
    other = Client(OTHER_CHANNEL, launched_engine=True)
    try:
        mine = await ipc_client.request(1)
        theirs = await other.request(1)
        requests = {
            r["response_channel_id"]: r
            for r in (stand_in.receive_request(), stand_in.receive_request())
        }

        await asyncio.to_thread(stand_in.answer, requests[CHANNEL], 40)
        await asyncio.to_thread(stand_in.answer, requests[OTHER_CHANNEL], 60)

        assert await numbers_received(mine) == list(range(40))
        assert await numbers_received(theirs) == list(range(60))
        assert mine.empty() and theirs.empty()
        # Each client was answered over an endpoint of its own.
        assert sorted(stand_in.routes) == [CHANNEL, OTHER_CHANNEL]
    finally:
        other.close()


@pytest.mark.parametrize(
    ("channel_id", "name"),
    [
        (0x1A2B3C, "pie_response_1a2b3c.ipc"),
        # No padding, and a leading zero nibble is not printed.
        (0x0ABC, "pie_response_abc.ipc"),
        # Lowercase, all 16 digits of a (pid << 32) | random id.
        (0xFFFFFFFF00000001, "pie_response_ffffffff00000001.ipc"),
        (0xDEADBEEFCAFEF00D, "pie_response_deadbeefcafef00d.ipc"),
    ],
)
def test_the_endpoint_has_the_name_the_engine_dials(ipc_root, channel_id, name):
    """Literal names, because the stand-in engine above dials whatever
    response_pull_path returns. PIE formats "{}{:x}{}" with "pie_response_" and
    ".ipc" (transport.cpp get_response_route, ipc/constants.hpp); a client that
    listens under any other name is never dialled and every request times out."""
    assert endpoints.response_pull_path(channel_id) == ipc_root / name


def test_closing_the_sockets_removes_the_endpoint(ipc_client):
    endpoint = endpoints.response_pull_path(CHANNEL)
    assert endpoint.exists()

    ipc_client.close()

    assert ipc_client.state.response_pull_socket is None
    assert not endpoint.exists()


def test_the_endpoint_is_removed_when_the_close_is_skipped(ipc_client):
    endpoint = endpoints.response_pull_path(CHANNEL)
    assert endpoint.exists()

    # All that runs at interpreter exit, where closing NNG sockets can deadlock.
    InferenceEngine._request_dispatcher_shutdown(ipc_client.ctx)

    assert ipc_client.state.response_pull_socket is not None
    assert not endpoint.exists()

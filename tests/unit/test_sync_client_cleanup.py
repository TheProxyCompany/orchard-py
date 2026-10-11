import asyncio
import concurrent.futures
import threading

import pytest

from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import ModelRegistry
from orchard.clients.client import Client
from orchard.engine.global_context import GlobalContext


@pytest.fixture
def client():
    context = GlobalContext()
    ipc = IPCState(context)
    client = Client(ipc, ModelRegistry(ipc))
    try:
        yield client
    finally:
        client.close()


def test_sync_delta_timeout_closes_the_pending_generator(client, monkeypatch):
    monkeypatch.setenv("ORCHARD_SYNC_DELTA_TIMEOUT_S", "0.02")
    finished = threading.Event()

    async def deltas():
        try:
            yield "first"
            await asyncio.Queue().get()
        finally:
            await asyncio.sleep(0)
            finished.set()

    client._start_sync_event_loop()
    stream = client._sync_iterator_bridge(deltas())
    assert next(stream) == "first"
    with pytest.raises(RuntimeError, match="Timed out waiting for inference delta"):
        next(stream)
    assert finished.wait(1), "timed-out stream kept its pending read alive"
    client.close()
    assert not client._sync_thread.is_alive()
    assert client._sync_loop.is_closed()


def test_close_drains_an_active_sync_waiter(client, monkeypatch):
    monkeypatch.setenv("ORCHARD_SYNC_DELTA_TIMEOUT_S", "0.2")
    started = threading.Event()
    finished = threading.Event()
    errors = []

    async def deltas():
        try:
            started.set()
            yield await asyncio.Queue().get()
        finally:
            await asyncio.sleep(0)
            finished.set()

    client._start_sync_event_loop()
    stream = client._sync_iterator_bridge(deltas())

    def consume():
        try:
            next(stream)
        except (concurrent.futures.CancelledError, RuntimeError) as error:
            errors.append(error)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert started.wait(1)
    client.close()
    consumer.join(timeout=1)
    assert not consumer.is_alive(), "closing the client left a sync caller waiting"
    assert finished.is_set(), "closing the loop abandoned the generator's cleanup"
    assert len(errors) == 1
    assert isinstance(errors[0], concurrent.futures.CancelledError)
    assert client._sync_loop.is_closed()


def test_completed_sync_stream_closes_and_client_can_restart(client):
    async def deltas():
        yield "first"
        yield "second"

    for _ in range(2):
        client._start_sync_event_loop()
        assert list(client._sync_iterator_bridge(deltas())) == ["first", "second"]
        client.close()
        assert not client._sync_thread.is_alive()
        assert client._sync_loop.is_closed()

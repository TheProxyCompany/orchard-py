import json
import os
import subprocess
import sys
from pathlib import Path

from orchard.ipc import endpoints


def test_short_ipc_root_is_preserved():
    candidate = Path("/tmp") / f"orchard-short-{os.getpid()}"

    assert endpoints._bounded_ipc_root(candidate) == candidate.resolve()


def test_fnv1a64_matches_cross_language_fixture():
    assert endpoints._fnv1a64(b"/deterministic/orchard/ipc") == 0x9262F36FFEAE1C45


def test_long_ipc_root_is_compacted_deterministically(tmp_path):
    candidate = tmp_path / ("cache-" + "x" * 140) / "ipc"

    first = endpoints._bounded_ipc_root(candidate)
    second = endpoints._bounded_ipc_root(candidate)

    assert first == second
    assert first != candidate.resolve()
    assert first.name.startswith(f"orchard-ipc-{os.getuid()}-")
    assert endpoints._socket_path_fits(
        first / endpoints._LONGEST_SOCKET_NAME
    )


def test_distinct_long_ipc_roots_do_not_alias(tmp_path):
    first = tmp_path / ("a" * 140) / "ipc"
    second = tmp_path / ("b" * 140) / "ipc"

    assert endpoints._bounded_ipc_root(first) != endpoints._bounded_ipc_root(second)


def test_long_cache_root_exports_bindable_ipc_root(tmp_path):
    cache_root = tmp_path / ("cache-" + "x" * 140)
    env = os.environ.copy()
    env.pop("ORCHARD_IPC_ROOT", None)
    env["ORCHARD_CACHE_ROOT"] = str(cache_root)
    script = """
import json
import os
import pynng
from orchard.ipc.endpoints import IPC_ROOT, response_route_path, response_route_url

channel_id = (1 << 64) - 1
with pynng.Pull0(listen=response_route_url(channel_id)):
    print(json.dumps({
        "ipc_root": str(IPC_ROOT),
        "exported_root": os.environ["ORCHARD_IPC_ROOT"],
        "route_path_bytes": len(os.fsencode(response_route_path(channel_id))),
    }))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ipc_root"] == payload["exported_root"]
    assert payload["route_path_bytes"] <= endpoints._IPC_SOCKET_PATH_MAX_BYTES

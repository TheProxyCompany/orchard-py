import importlib.util
import os
import socket
import stat
import tempfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "orchard/ipc/endpoints.py"
LONGEST_SOCKET = "pie_response_ffffffffffffffff.ipc"


def compact_root(candidate, *, canonicalized=False):
    digest = 0xCBF29CE484222325
    for byte in os.fsencode(candidate if canonicalized else candidate.resolve()):
        digest = ((digest ^ byte) * 0x100000001B3) % (1 << 64)
    return Path("/tmp").resolve() / f"orchard-ipc-{os.getuid()}-{digest:016x}"


@pytest.fixture
def roots(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="ipc-py-", dir="/tmp") as directory:
        root = Path(directory).resolve()
        monkeypatch.chdir(root)
        monkeypatch.setenv("HOME", str(root))
        for key in ("ORCHARD_IPC_ROOT", "ORCHARD_CACHE_ROOT", "XDG_CACHE_HOME"):
            monkeypatch.delenv(key, raising=False)
        yield root


@pytest.fixture
def load_endpoints(roots):
    created = set()
    temporary = Path("/tmp").resolve()
    existing = set(temporary.glob(f"orchard-ipc-{os.getuid()}-*"))

    def load(candidate):
        compact = compact_root(candidate)
        assert not compact.exists(), "test namespace already exists"
        created.add(compact)
        spec = importlib.util.spec_from_file_location("isolated_ipc_endpoints", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if (
            module.IPC_ROOT.parent == temporary
            and module.IPC_ROOT.name.startswith(f"orchard-ipc-{os.getuid()}-")
            and module.IPC_ROOT not in existing
        ):
            created.add(module.IPC_ROOT)
        return module

    yield load
    for path in created:
        if path.exists():
            path.rmdir()


def test_short_override_and_endpoint_urls(roots, monkeypatch, load_endpoints):
    candidate = roots / "socket"
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    monkeypatch.setenv("ORCHARD_CACHE_ROOT", str(roots / "ignored"))
    module = load_endpoints(candidate)
    assert module.IPC_ROOT == candidate
    assert module.REQUEST_URL == f"ipc://{candidate}/pie_requests.ipc"
    assert module.RESPONSE_URL == f"ipc://{candidate}/pie_responses.ipc"
    assert module.MANAGEMENT_URL == f"ipc://{candidate}/pie_management.ipc"


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_exact_socket_path_boundary(roots, monkeypatch, load_endpoints, extra_bytes):
    length = 103 - len(os.fsencode(roots)) - len(LONGEST_SOCKET) - 2 + extra_bytes
    candidate = roots / ("x" * length)
    assert len(os.fsencode(candidate / LONGEST_SOCKET)) == 103 + extra_bytes
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    module = load_endpoints(candidate)
    assert module.IPC_ROOT == (
        candidate if extra_bytes == 0 else compact_root(candidate)
    )
    assert len(os.fsencode(module.IPC_ROOT / LONGEST_SOCKET)) <= 103


@pytest.mark.parametrize("setting", ["ORCHARD_IPC_ROOT", "ORCHARD_CACHE_ROOT"])
def test_long_override_is_compacted(roots, monkeypatch, load_endpoints, setting):
    override = roots / ("long-" * 20)
    candidate = override if setting == "ORCHARD_IPC_ROOT" else override / "ipc"
    monkeypatch.setenv(setting, str(override))
    module = load_endpoints(candidate)
    assert module.IPC_ROOT == compact_root(candidate)
    assert module.IPC_ROOT.is_dir()
    assert stat.S_IMODE(module.IPC_ROOT.stat().st_mode) == 0o700


def test_byte_length_not_character_count(roots, monkeypatch, load_endpoints):
    candidate = roots / ("\u00e9" * 25)
    assert len(str(candidate / LONGEST_SOCKET)) <= 103
    assert len(os.fsencode(candidate / LONGEST_SOCKET)) > 103
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    assert load_endpoints(candidate).IPC_ROOT == compact_root(candidate)


def test_symlink_prefix_and_missing_suffix(roots, monkeypatch, load_endpoints):
    real = roots / "real"
    real.mkdir()
    (roots / "alias").symlink_to(real, target_is_directory=True)
    candidate = roots / "alias" / ("x" * 80)
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    assert load_endpoints(candidate).IPC_ROOT == compact_root(real / ("x" * 80))


def test_relative_root_and_literal_tilde(roots, monkeypatch, load_endpoints):
    candidate = roots / "~" / "socket"
    monkeypatch.setenv("ORCHARD_IPC_ROOT", "~/socket")
    assert load_endpoints(candidate).IPC_ROOT == candidate


@pytest.mark.parametrize("dangling", [False, True])
def test_weak_canonicalization_stops_at_missing_prefix(
    roots, monkeypatch, load_endpoints, dangling
):
    bootstrap = roots / "bootstrap"
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(bootstrap))
    module = load_endpoints(bootstrap)
    if not dangling:
        (roots / "real").mkdir()
    (roots / "alias").symlink_to(roots / "real", target_is_directory=True)
    suffix = "x" * 90
    candidate = (
        roots / "alias" / suffix
        if dangling
        else roots / "missing" / ".." / "alias" / suffix
    )
    expected = roots / "alias" / suffix
    assert module._canonical_path(candidate) == expected
    assert module._bounded_ipc_root(candidate) == compact_root(
        expected, canonicalized=True
    )


def test_empty_ipc_override_uses_cache(roots, monkeypatch, load_endpoints):
    candidate = roots / "cache" / "ipc"
    monkeypatch.setenv("ORCHARD_IPC_ROOT", "")
    monkeypatch.setenv("ORCHARD_CACHE_ROOT", str(roots / "cache"))
    assert load_endpoints(candidate).IPC_ROOT == candidate


def test_cache_directory_exists_before_path_projection(
    roots, monkeypatch, load_endpoints
):
    (roots / "real").mkdir()
    (roots / "alias").symlink_to(roots / "real", target_is_directory=True)
    cache = roots / "missing" / ".." / "alias" / ("x" * 90)
    candidate = cache / "ipc"
    monkeypatch.setenv("ORCHARD_CACHE_ROOT", str(cache))
    assert load_endpoints(candidate).IPC_ROOT == compact_root(candidate)


@pytest.mark.parametrize(
    "platform,xdg", [("darwin", False), ("linux", False), ("linux", True)]
)
def test_platform_default(roots, monkeypatch, load_endpoints, platform, xdg):
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setenv("ORCHARD_IPC_ROOT", "")
    monkeypatch.setenv("ORCHARD_CACHE_ROOT", "")
    if xdg:
        monkeypatch.setenv("XDG_CACHE_HOME", str(roots / "xdg"))
        base = roots / "xdg"
    else:
        base = roots / ("Library/Caches" if platform == "darwin" else ".cache")
    candidate = base / "com.theproxycompany/ipc"
    expected = (
        candidate
        if len(os.fsencode(candidate / LONGEST_SOCKET)) <= 103
        else compact_root(candidate)
    )
    assert load_endpoints(candidate).IPC_ROOT == expected


def test_existing_directory_permissions_are_preserved(
    roots, monkeypatch, load_endpoints
):
    candidate = roots / "existing"
    candidate.mkdir(mode=0o755)
    candidate.chmod(0o755)
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    assert load_endpoints(candidate).IPC_ROOT == candidate
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o755


def test_compacted_path_supports_longest_socket_name(
    roots, monkeypatch, load_endpoints
):
    candidate = roots / ("x" * 100)
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    module = load_endpoints(candidate)
    address = module.IPC_ROOT / LONGEST_SOCKET
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(address))
            server.listen(1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect(str(address))
                connection, _ = server.accept()
                with connection:
                    connection.sendall(b"ipc-parity")
                    assert client.recv(32) == b"ipc-parity"
    finally:
        address.unlink(missing_ok=True)


def test_existing_file_is_not_replaced(roots, monkeypatch, load_endpoints):
    candidate = roots / "existing-file"
    candidate.write_text("preserve")
    monkeypatch.setenv("ORCHARD_IPC_ROOT", str(candidate))
    with pytest.raises(FileExistsError):
        load_endpoints(candidate)
    assert candidate.read_text() == "preserve"

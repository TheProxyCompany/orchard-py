"""Orchard engine binary fetching and installation."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import sys
import tarfile
import threading
from pathlib import Path

import dotenv
import requests
from filelock import FileLock

dotenv.load_dotenv()

MANIFEST_URL = "https://prod.proxy.ing/functions/v1/get-release-manifest"
DEFAULT_CHANNEL = "stable"
ORCHARD_HOME = Path.home() / ".orchard"

REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 600
MAX_RETRIES = 3

# Cached update check result
_update_available: str | None = None
_update_check_done = threading.Event()


class FetchError(Exception):
    """Base exception for fetch operations."""


class ManifestError(FetchError):
    """Failed to fetch or parse release manifest."""


class DownloadError(FetchError):
    """Failed to download engine binary."""


class IntegrityError(FetchError):
    """Downloaded file failed integrity check."""


class ExtractionError(FetchError):
    """Failed to extract downloaded archive."""


def get_engine_path() -> Path:
    """Return path to the engine binary, downloading if necessary."""
    # Local dev override takes priority if it exists
    local_build = os.environ.get("PIE_LOCAL_BUILD")
    if local_build:
        local_path = Path(local_build) / "bin" / "proxy_inference_engine"
        if local_path.exists():
            return local_path
        # Fall through to download if local build not found

    binary_path = ORCHARD_HOME / "bin" / "proxy_inference_engine"

    if binary_path.exists():
        return binary_path

    # Acquire lock to prevent concurrent downloads
    ORCHARD_HOME.mkdir(parents=True, exist_ok=True)
    lock_path = ORCHARD_HOME / "install.lock"

    with FileLock(str(lock_path), timeout=300):
        # Another process may have installed while we waited
        if binary_path.exists():
            return binary_path

        _print_status("Orchard engine not found")
        download_engine()

    if not binary_path.exists():
        raise FetchError("Download completed but binary not found")

    return binary_path


def download_engine(
    channel: str = DEFAULT_CHANNEL,
    version: str | None = None,
) -> None:
    """Download and install the engine binary."""
    manifest = _fetch_manifest(channel)

    if version is None:
        version = manifest.get("latest")
        if not version:
            raise ManifestError(
                f"No latest version defined for \033[32m{channel}\033[0m channel"
            )

    versions = manifest.get("versions", {})
    if version not in versions:
        available = ", ".join(sorted(versions.keys())) or "(none)"
        raise ManifestError(
            f"Version \033[32m{version}\033[0m not found in \033[32m{channel}\033[0m channel.\n"
            f"  Available versions: {available}"
        )

    info = versions[version]
    url = info.get("url")
    expected_sha256 = info.get("sha256")

    if not url:
        raise ManifestError(f"No download URL for version \033[32m{version}\033[0m")

    _print_status(f"Downloading \033[32m{version}\033[0m")
    content = _download_with_progress(url, expected_sha256)

    _print_status("Extracting")
    _extract_and_install(content, version)

    _print_status(f"Installed \033[32m{version}\033[0m", done=True)


def get_installed_version() -> str | None:
    """Return currently installed version, or None if not installed."""
    version_file = ORCHARD_HOME / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip() or None
    return None


def check_for_updates(channel: str = DEFAULT_CHANNEL) -> str | None:
    """Return latest version if an update is available, else None."""
    installed = get_installed_version()
    if not installed:
        return None

    try:
        manifest = _fetch_manifest(channel)
        latest = manifest.get("latest")
        if latest and latest != installed:
            return latest
    except FetchError:
        pass  # Silently ignore update check failures

    return None


def check_for_updates_async(channel: str = DEFAULT_CHANNEL) -> None:
    """Fire-and-forget background update check with telemetry."""
    thread = threading.Thread(
        target=_background_update_check,
        args=(channel,),
        daemon=True,
    )
    thread.start()


def get_available_update() -> str | None:
    """Return cached update version if available, None otherwise. Non-blocking."""
    if _update_check_done.is_set():
        return _update_available
    return None


def _background_update_check(channel: str) -> None:
    """Background thread target for update checking."""
    global _update_available
    try:
        _update_available = check_for_updates(channel)
    except Exception:
        pass  # Never crash the background thread
    finally:
        _update_check_done.set()


def _fetch_manifest(channel: str) -> dict:
    """Fetch the release manifest from the server."""
    params = {
        "channel": channel,
        "v": get_installed_version() or "unknown",
        "os": platform.system().lower(),
        "arch": platform.machine(),
    }

    try:
        resp = requests.get(MANIFEST_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout as e:
        raise ManifestError("Timed out fetching release manifest") from e
    except requests.exceptions.ConnectionError as e:
        raise ManifestError(
            "Could not connect to release server.\n"
            "  Check your internet connection and try again."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise ManifestError(f"Server returned {e.response.status_code}") from e
    except requests.exceptions.JSONDecodeError as e:
        raise ManifestError("Invalid manifest format from server") from e


def _download_with_progress(url: str, expected_sha256: str | None) -> bytes:
    """Download file with progress bar and integrity verification."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            chunks: list[bytes] = []
            downloaded = 0

            for chunk in resp.iter_content(chunk_size=8192):
                chunks.append(chunk)
                downloaded += len(chunk)
                if total:
                    _print_progress(downloaded, total)

            if total:
                _clear_progress()

            content = b"".join(chunks)

            if expected_sha256:
                actual = hashlib.sha256(content).hexdigest()
                if actual != expected_sha256:
                    raise IntegrityError(
                        f"SHA256 verification failed.\n"
                        f"  Expected: {expected_sha256}\n"
                        f"  Got:      {actual}\n"
                        f"  This could indicate a corrupted download or tampering."
                    )

            return content

        except requests.exceptions.Timeout as e:
            if attempt == MAX_RETRIES - 1:
                raise DownloadError("Download timed out after multiple attempts") from e
            _print_status(f"Timeout, retrying ({attempt + 2}/{MAX_RETRIES})")

        except requests.exceptions.ConnectionError as e:
            if attempt == MAX_RETRIES - 1:
                raise DownloadError(
                    "Connection failed.\n"
                    "  Check your internet connection and try again."
                ) from e
            _print_status(f"Connection lost, retrying ({attempt + 2}/{MAX_RETRIES})")

        except requests.exceptions.HTTPError as e:
            raise DownloadError(
                f"Download failed: HTTP {e.response.status_code}"
            ) from e

    raise DownloadError("Download failed after multiple attempts")


def _extract_and_install(content: bytes, version: str) -> None:
    """Extract tarball and set up the installation."""
    ORCHARD_HOME.mkdir(parents=True, exist_ok=True)
    bin_dir = ORCHARD_HOME / "bin"

    # Clean existing bin directory for atomic update
    if bin_dir.exists():
        shutil.rmtree(bin_dir)

    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            # Security: validate paths before extraction
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ExtractionError(f"Unsafe path in archive: {member.name}")
            tar.extractall(ORCHARD_HOME)
    except tarfile.TarError as e:
        raise ExtractionError("Failed to extract archive") from e

    binary_path = bin_dir / "proxy_inference_engine"
    if not binary_path.exists():
        raise ExtractionError("Archive did not contain expected binary")

    binary_path.chmod(0o755)

    version_file = ORCHARD_HOME / "version.txt"
    version_file.write_text(version)


def _print_status(message: str, done: bool = False) -> None:
    """Print a status message."""
    prefix = "\033[32m✓\033[0m" if done else "\033[34m→\033[0m"
    print(f"{prefix} {message}")


def _print_progress(current: int, total: int) -> None:
    """Print download progress bar."""
    width = 40
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    mb_current = current / (1024 * 1024)
    mb_total = total / (1024 * 1024)
    sys.stdout.write(f"\r  [{bar}] {mb_current:.1f}/{mb_total:.1f} MB")
    sys.stdout.flush()


def _clear_progress() -> None:
    """Clear the progress bar line."""
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

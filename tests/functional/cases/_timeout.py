import os

from tests.shared_owner import shared_owner_enabled

# The combined proof owns one bounded process-level watchdog and does not put a
# second timeout on queued inference. Standalone cases retain a useful local
# failure bound.
HTTP_TIMEOUT_S: float | None = (
    None
    if shared_owner_enabled()
    else float(os.getenv("ORCHARD_TEST_HTTP_TIMEOUT_S", "180"))
)

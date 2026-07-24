import os

# Per-request HTTP timeout for functional cases. Solo runs finish well inside
# the 180s default; the uncapped buckshot volley (width 21, every suite
# surviving) legitimately queues single requests for 300s+, so the buckshot
# runner sizes this up via ORCHARD_TEST_HTTP_TIMEOUT_S instead of letting the
# client cap masquerade as an engine failure.
HTTP_TIMEOUT_S = float(os.getenv("ORCHARD_TEST_HTTP_TIMEOUT_S", "180"))

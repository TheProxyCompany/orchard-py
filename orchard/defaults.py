import os

MAX_GENERATED_TOKENS = 8192

# Ceiling on the wait for the next engine delta before a request 502s.
# 30s suits interactive serving; storm harnesses (buckshot, concurrent
# suite volleys) queue first tokens far past it while schedulers contend,
# so they raise it via the environment instead of failing on latency.
DELTA_TIMEOUT_S = float(os.getenv("ORCHARD_DELTA_TIMEOUT_S", "30"))

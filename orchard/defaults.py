import os

MAX_GENERATED_TOKENS = 8192

# Tolerated engine-wide delta silence while a request waits for its next
# delta. This is a stall detector, not a latency budget: any delta or model
# activation event from the engine — for any request — resets it. A wedged
# engine (nothing flowing to anyone) still fails requests after 30s.
DELTA_TIMEOUT_S = float(os.getenv("ORCHARD_DELTA_TIMEOUT_S", "30"))

# Ceiling on one delta wait even while the engine is demonstrably busy with
# other requests. Bounds the case of a request the engine dropped. Sized from
# the joint buckshot, where 18 models take turns in memory that holds a few
# of them: a request waited up to 512s for its model to come back and then
# 279s to decode beside a diffusion model (64 GiB lane, root run 45011140250,
# 2026-11-22). 900s clears that 791s by about 109s.
DELTA_HARD_TIMEOUT_S = float(os.getenv("ORCHARD_DELTA_HARD_TIMEOUT_S", "900"))

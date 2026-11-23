import os

MAX_GENERATED_TOKENS = 8192

# Tolerated engine-wide delta silence while a request waits for its next
# delta. This is a stall detector, not a latency budget: any delta or model
# activation event from the engine — for any request — resets it. A wedged
# engine (nothing flowing to anyone) still fails requests after 30s.
DELTA_TIMEOUT_S = float(os.getenv("ORCHARD_DELTA_TIMEOUT_S", "30"))

# Ceiling on one delta wait even while the engine is demonstrably busy with
# other requests. Bounds the case of a request the engine dropped. Sized from
# the width-21 buckshot cold peak: ~250 batched requests landing at once fan
# out to ~800 sequences, and the deepest first prefill starts 91s after
# submit (pt1 py6_engine.log, 2026-07-23) — 300s is ~3x that tail.
DELTA_HARD_TIMEOUT_S = float(os.getenv("ORCHARD_DELTA_HARD_TIMEOUT_S", "300"))

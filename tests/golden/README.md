# Golden path

## NOTES (greedy decoding: the suite is red until re-recorded)

Every case now decodes greedily (see "Sampling"). The files under `data/` were
recorded with seeded sampling at each model's recommended temperature, so they
no longer describe what the suite asks the engine for: **expect this suite to
be red until each model is re-recorded against the final engine build.** The
one exception should be `granite_switch`, whose recommended temperature already
was 0, so its requests are unchanged.

Re-record one model at a time, with only that model loaded, each in its own
fresh engine namespace, and review the git diff of `data/<template_type>/`
after each run (a run writes nothing unless every case of that model passes):

    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh llama3
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh gemma4
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh qwen3_5
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh moondream3
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh afmoe
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh lfm2_5
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh olmo_hybrid
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh nemotron_h
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh granite_switch
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh gpt_oss
    ORCHARD_CACHE_ROOT=<fresh dir> PIE_LOCAL_BUILD=<engine release dir> scripts/record_golden.sh pipeline

`pipeline` goes last and is the only run that needs several models at once:
`gemma4` and `moondream3` preloaded, then the test hydrates, one at a time,
`ideogram-ai/ideogram-4-fp8`, `black-forest-labs/FLUX.2-klein-4B`, the two
Qwen3-TTS, Parakeet and the two Qwen3-ASR checkpoints, and the image-edit cases
load `Qwen/Qwen-Image-Edit` on demand. It rewrites only `data/gemma4/image_*`
and `data/moondream3/image_*` (the audio case records nothing).

Two things to watch while re-recording. Greedy was dropped once before
(4c7400c, July), whose message says qwen3_5's golden "ran away unbounded" under
it. The model's recommended penalties still apply under greedy now (they did
not then: temperature 0 rode on the default lane, which has none), but nothing
here was run against an engine. If a thinking model loops, its cases fail,
nothing is written, and that is a finding to bring back rather than record
around. And
`llama3/tool_chaining/turn3` carries two blessed per-device variants; a
re-record leaves one, so the other device family needs `GOLDEN_ADD_VARIANT=1`
again if it still resolves a near-tie differently.

## What this is

Multi-turn agentic rollouts that certify the engine's behavior doesn't **drift**
over time. Where `functional/` checks one capability's contract in isolation,
golden tests run composed flows (reason → tool → continue) and pin exact behavior:
each turn's normalized event stream is compared with its recording in
`data/<template_type>/<scenario>.json` (`golden_io.py`).

Each scenario is a flat file under `cases/` taking `client` (+ `model`), gated
inline by capability (`if not model.tools: return`). `model` ids are the
architecture (`llama3`, `gpt_oss`, …) since what we certify is the arch in the
engine. Cases never run on their own: `cases/registry.py` collects them and
runs all of a model's cases concurrently, from `test_matrix.py` (one test per
model, plus one for the multi-model pipeline cases) and from the buckshot volley.

## Sampling

The cases pass `deterministic=True` and no temperature. On its own that is
**not** greedy: the client (`orchard/clients/client.py`, request building)
fills omitted sampling fields from the profile's `recommended` lane and pins
`rng_seed=11`, so gpt-oss and gemma-4 would sample at temperature 1.0 and the
recorded token would be `argmax(logprob/T + seeded noise)`, routinely decided by
less than one bf16 logit step. The runner therefore hands every case a
`GreedyClient` (`cases/registry.py`) that pins `temperature=0.0` and
`deterministic=True` on every Responses call and on its rendered-prompt
preview, whatever the case passes. At temperature 0 the engine takes the plain
argmax and skips top-p/top-k/min-p; the profile's penalties still apply;
`deterministic` keeps the engine's reproducible scheduling on. Image and audio
generation calls are not touched (TTS keeps its own temperature and seed).

## Reading a failure

A drift report names the turn, both event counts, the index of the first
differing event, the last 60 characters both runs agreed on, and the two texts
(or, when the text is the same, the fields that differ):

    golden drift gpt_oss/multi_tool/turn1: event count: golden=135 live=109; first diff at index 76 after 'co". For Tokyo time, call get_time with timezone "Asia/Tokyo':
      golden: response.output_token '".\n\n'
      live:   response.output_token '".'

A drifted turn does not stop its case: later turns still run and are compared,
and every drifted turn is reported when the case ends (one drift used to hide
the next). What fails is unchanged: any drifted turn fails the case, and so does
any semantic assertion, which is then shown as the cause above the drift report.
A later turn's prompt is built from the live earlier turn, so a later drift is
independent only if the earlier turn's visible output (tool calls, answer) was
unchanged.

A failing golden test means behavior changed — suspect the engine, not the
assertion. Keep expectations strict; re-baseline only on intended changes.

## Recording

- A turn with no recording is staged and written only if its whole test passes
  (`conftest.py`), so a failing test never leaves a buggy golden behind. Buckshot
  treats a missing recording as a failure.
- `GOLDEN_RECORD=1` re-records: every turn is staged instead of asserted and a
  passing test replaces each file it staged. `GOLDEN_ADD_VARIANT=1` instead adds
  the live stream as one more blessed variant of a drifted turn (per-device
  near-ties).
- `ORCHARD_TEST_MODELS=<template_type>[,…]` narrows the matrix (`tests/models.py`)
  for one session: the engine fixture preloads only those checkpoints and the
  matrix tests parametrize over only them. `scripts/record_golden.sh` combines
  the two and selects only that model's test, and refuses to run without
  `PIE_LOCAL_BUILD` and `ORCHARD_CACHE_ROOT`.
- The human is in the loop at the git diff: bless a recording by committing it.

## Scenarios

Per model (`model_cases`):

- **`reason_then_tool`** — [tools] reason → call the right tool with the right
  args → feed the result → grounded answer. Pins the event lifecycle (one
  reasoning block, delta accumulation, one tool call) and the semantic outcome.
- **`reason_then_structured`** — [reasoning] reason, then emit strict
  `json_schema`. Reasoning terminates cleanly; the object parses to the exact
  expected value. (The Gemma-harmony / PSE non-termination class.)
- **`tool_selection`** — [tools] 18 tools offered; the model picks the one correct
  tool with the right args.
- **`multi_tool`** — [tools] a prompt needing two tools; both called with correct
  args (in one turn or split across turns) and both results in the answer.
- **`tool_chaining`** — [tools] dependent tools: `find_key` → `unlock_chest` with
  that exact key → answer grounded in the chest contents.
- **`tool_result_grounding`** — [tools] the final answer must contain the injected
  tool value (9°C, snowing), not the model's prior.
- **`thinking_on_off`** — [reasoning, switchable] same prompt with reasoning
  enabled vs suppressed: reasoning present when on, **zero** reasoning tokens
  when off, correct answer both ways.

Pipeline (`pipeline_cases`):

- **`image_tool_result_grounding`** — gemma4 calls an image tool, Ideogram or
  FLUX generates (and Qwen-Image-Edit edits), moondream3 verifies blind.
- **`audio_telephone`** — TTS → speech-to-text round trip; exact transcripts, no
  recording.

Not built: batch invariance (same scenario at batch sizes 1/2/4/8), token
identity across turns, vision multi-turn.

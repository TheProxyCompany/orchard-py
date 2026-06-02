# Golden path

Multi-turn agentic rollouts that certify the engine's behavior doesn't **drift**
over time. Where `functional/` checks one capability's contract in isolation,
golden tests run composed flows (reason → tool → continue) and pin exact behavior.

Run like any functional test. Each scenario is a flat file taking the `client` +
`model` fixtures, gated inline by capability (`if not model.tools: return`)
and driven at temperature 0 / deterministic. `model` ids are the architecture
(`llama3`, `gpt_oss`, …) since what we certify is the arch in the engine.

## Scenarios

Implemented:

- **`test_reason_then_tool.py`** — [tools] reason → call the right tool with the
  right args → feed the result → grounded answer. The streaming pattern: pins the
  event lifecycle (one reasoning block, delta accumulation, one tool call) and the
  semantic outcome.

Scoped (to build):

1. **`test_reason_then_structured.py`** — [reasoning ∩ structured] reason, then emit
   strict `json_schema`. Reasoning terminates cleanly; the object parses to the exact
   expected value. (The Gemma-harmony / PSE non-termination class.)
2. **`test_tool_selection.py`** — [tools] ~18 tools offered; the model picks the one
   correct tool (not the other 17) with the right args.
3. **`test_multi_tool.py`** — [tools] a prompt needing two tools; both called with
   correct args and both results integrated into the answer.
4. **`test_batch_invariance.py`** — [all] the same scenario at batch sizes 1/2/4/8;
   every batch position is identical and matches the size-1 baseline.
5. **`test_token_identity.py`** — [all] across a reason → tool → answer rollout, turn-2
   prompt token ids start with turn-1 prompt+completion ids byte-for-byte (the
   extension property). Written as the spec; **xfail** until an append-raw-ids bridge
   replaces re-rendering history.
6. **`test_tool_result_grounding.py`** — [tools] the final answer must contain the
   injected tool value (e.g. `65`/`foggy`), not a hallucinated one.
7. **`test_vision_multiturn.py`** — [vision] image + question, then a follow-up about
   the same image (image persists in history across turns).
8. **`test_thinking_on_off.py`** — [reasoning] same prompt with reasoning enabled vs
   suppressed: reasoning present when on, **zero** reasoning tokens when off, correct
   answer both ways.

A failing golden test means behavior changed — suspect the engine, not the
assertion. Keep expectations strict; re-baseline only on intended changes.

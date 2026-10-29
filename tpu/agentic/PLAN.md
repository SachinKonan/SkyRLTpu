# Agentic no-train arms: build plan (2026-10-28)

Two arms that replace the no-train scaffold's single-call improvement operator with an agent episode, compared with
the old 16x32 no-train runs and RL on FLOPs (Kaplan: 2N per processed token, 6N per trained token) and grader calls.

## Arms

| | A: agentic scaffold | B: agent teams |
|---|---|---|
| step | PUCT samples 16 states; 16 agents per state | 16 teams x 16 agents, one message board per team |
| agent starts from | parent code in the prompt | the team board |
| grading | each submission graded once (256/step) | same |
| after grading | top 2 of each group of 16 into the tree | one ranking post per team: rank, id `t{turn}.{rank}`, score, change vs previous team best; code stored |
| memory | PUCT tree | board (persists; contexts flushed every turn) |

Episode: 26k working tokens (every generated token, compaction summaries included), then a forced 6k answer phase.
B spends a fixed communication slice (start: 4k) at the start of the 26k with board tools and `read` only.

## Tools (native shell, search and web tools off)

Native names and schemas per harness, with the path parameter removed. The working solution is a hidden per-agent
file that starts empty.

| tool | dsh (Qwen) | OpenCode (Muse) | errors |
|---|---|---|---|
| read | `read(offset?, limit?)` | `read(offset?, limit?)` | nothing written yet |
| write | `write(content)` | `write(content)` | |
| edit | `edit(old_string, new_string, replace_all?)` | `edit(oldString, newString, replaceAll?)` | nothing written; no or ambiguous match |
| compile | `compile()`: static check only, never executes | same | nothing written |
| submit | `submit(sources=[ids])`: ends the episode | same | nothing written; bad source id |
| B | `board_read(scroll=0)`, `board_read_code(id, scroll=0)`: newest first, K=2048 tokens per window (model tokenizer); `board_post(text)` capped ~1k tokens; live within a turn | same | |

At the end of the answer phase an unsubmitted written solution is auto-submitted; nothing written means no child.

## Components

1. **Gateway** (OpenAI-compatible `/v1/chat/completions` in front of the run's ingress, `TTD_FROZEN_BASE_URL`).
   The farms are vLLM's OpenAI server without a tool-call parser, so the gateway renders chat + tools with the
   model's HF chat template, calls `/v1/completions` (native `thinking_token_budget`, as `tpu/simpletes/backend.py`),
   and parses tool calls itself:
   - Qwen3.5/3.8: `<tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>`
     (Qwen3-Coder XML).
   - Muse: ATEM, `to=tool.fn` + `<atem:function_calls><atem:invoke name=...>`. Needs our own parser (biggest risk).
   - Per-session budget: counts generated tokens; at 26k switches the episode to the answer phase (6k cap, answer
     tools only); tags compaction requests; returns a standard context-overflow error if a request cannot fit.
   - Logs prompt and generated tokens per request and computes, per session, how much of each prompt repeats the
     previous request's tokens. That gives two hardware-independent FLOP counts: cache-ideal (new prompt tokens +
     generated) and no-cache (full prompt + generated). Prefix caching is on for v6e-8 and v5p-32 farms but
     deliberately off on every v4 farm (a cached greedy test on the v4 Qwen runtime diverged from fresh-cache runs,
     2026-09-19; `LORA_FARM.md`, `HYBRID_INFERENCE.md`). Agent turns on v4 re-prefill the whole context, roughly
     10x the cost, so run the agentic arms on v6e/v5p farms, or re-test caching on v4 first.
2. **Tool service** (Python): working files, compile, submit, program ids, boards, phase gating, ranking posts,
   provenance. One implementation; harness adapters are thin.
3. **Adapters**: a dsh plugin (`dsh --profile headless --json`) and OpenCode custom tools exposing the table above.
4. **Driver** `--driver agentic` in `tpu.swarm.ray_train.local_notrain`, next to `simpletes`: reuses the ingress,
   farm leases and grading transport; runs A (PUCT from the existing sampler) or B (boards), 256 episodes per step,
   grades, writes snapshots and metrics in the existing layout so the FLOP/metrics scripts work unchanged.
5. **Agent hosts: the farms.** A session is a harness process plus the tool service and never runs model-written
   code, so it needs ~0.5 core and ~1 GiB, mostly idle while waiting on the model. Add an `agent_session` family to
   the farms' grading core pool (small CPU/memory, long timeout). Run each A group / B team of 16 on one farm host,
   with a per-host gateway calling that farm's local vLLM engine: every turn of a session hits the same engine, so
   prefix caching works, and B's board can live on the host. The central driver (della-vis1) dispatches group/team
   tasks, collects submissions and token logs, grades through the existing transport, and writes ranking posts.
   Ship dsh (prebuilt, with Node.js) and the OpenCode binary as artifacts; no npm at runtime. v6e/v5p farms only
   while v4 caching is off.
   Placement (user, 2026-10-28): several groups per farm, ~8 groups/teams per farm, 2 farms per run. Sessions are
   pinned to one engine (cache locality). Qwen v5p-32 (4 engines, 64 seqs, ~17.7M KV tokens; 2026-10-07 numbers at
   22.5k ctx) holds 8 groups = 128 sessions x 32k: 2 groups per engine, 128 sessions sharing 64 slots. Qwen v6e-8
   (~1.6M KV) holds ~3 groups before cached prefixes are evicted.
   Gate: Qwen3.5/3.8 are hybrid (Gated DeltaNet) models; confirm the engines actually serve prefix-cache hits
   (vLLM prefix_cache_queries/hits metrics) before anything else. Earlier notes: untested on TPU for the hybrid
   runtime; the v4 test gave non-repeatable greedy outputs with caching on.

## Compaction (native, aligned)

- 32,768-token declared context for both harnesses; trigger ~80%; keep ~5k recent tokens; ~2k summary.
- dsh `@deepseek-ai/dsh-compaction-basic`: `auto: true`, `thresholdRatio: 0.8`, `headroomTokens: ~2048`
  (default 65536 would compact constantly at 32k), `retainTokens: ~5000`, `maxTokens: ~2048`.
- OpenCode `compaction`: `auto: true`, `buffer: ~6500`, `keep.tokens: ~5000`; model `limit.context: 32768`.
- Pin harness versions (dsh is a developer preview with breaking changes).

## Pilot

Erdős, one parent x 16 agents, Qwen in dsh: tokens per episode, cache-hit rate, turn latency, compaction and
forced-answer rates, scroll use. Then the Muse/OpenCode tool-call check, then full A and B runs (2 seeds each).

"""Cold-start probe driver: generate -> pre-gate -> judge -> metrics.

NO RL. No policy, no gradients, no training. This measures one thing: can a
BASE model produce trainable attempts at a Pallas kernel, before any RL is
built?

"Trainable" has a precise operational meaning here, and it is not the mean
score. ttt-discover computes advantage WITHIN a group of rollouts, so a
configuration in which every candidate in a group scores 0 contributes
exactly zero gradient no matter how good its mean looks. The headline metric
is therefore the **fraction of GROUP_SIZE-candidate groups with non-uniform
reward**.

Shape of the run, per round:
  * one grouped `/v1/completions` request per configuration (n=GROUP_SIZE,
    one shared prefill),
  * every generation immediately CPU pre-gated 16-way in parallel,
  * survivors submitted to the phase-5 queue for the real judge,
  * everything appended to a JSONL as it lands, so a teardown at any point
    still leaves a complete measurement of the rounds that finished.

Rounds are round-robin across configurations, so at whatever moment the wall
clock stops the run, every configuration has the SAME number of complete
groups and the comparison is still fair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tpu/

from pallas_arena.judge.queue import ArenaQueueClient  # noqa: E402
from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe import render as R  # noqa: E402
from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402
from pallas_arena.probe.prompts import get_prompt  # noqa: E402
from pallas_arena.probe.sampler import VllmSampler  # noqa: E402
from pallas_arena.probe.seam import SEAMS, compose, extract_fill  # noqa: E402

# gates the pre-gate can reach on its own; everything else needs a chip
PREGATE_TERMINAL = {"no_code", "ast", "exec", "poison_stub", "aot_export", "timeout", "rlimit", "harness"}


def sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class Probe:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.out.with_suffix(".jsonl").open("a", buffering=1)
        self.lock = threading.Lock()
        self.records: list[dict] = []
        self.t0 = time.time()
        self.deadline = self.t0 + args.wall_s

        self.configs = [
            c
            for c in C.all_configs(
                models=args.models.split(",") if args.models else None,
                variants=args.variants.split(",") if args.variants else None,
                tasks=args.tasks.split(",") if args.tasks else None,
            )
        ]
        self.samplers = {}
        for key in {c.model for c in self.configs}:
            spec = C.MODELS[key]
            url = dict(kv.split("=", 1) for kv in args.servers.split(",")).get(key)
            if not url:
                print(f"[probe] no server url for {key}; dropping its configurations", flush=True)
                continue
            self.samplers[key] = VllmSampler(url, spec.hf_id)
        self.configs = [c for c in self.configs if c.model in self.samplers]
        # probe_signatures returns (sigs, case_sig, adv_sig, grad_sig); the
        # export child wants ONLY the first element. Passing the whole tuple
        # made the child iterate (sigs, case_sig, ...) and do `sig["args"]`
        # on a LIST, so every candidate that was well-formed enough to REACH
        # export died at gate `aot_export` with
        #   TypeError: list indices must be integers or slices, not str
        # -- a harness fault wearing a candidate's clothes. It cost 14 of 32
        # arm-2 candidates before it was caught.
        self.sigs = {t: probe_signatures(t, C.TASK_CASES[t])[0] for t in {c.task for c in self.configs}}
        self.queue = ArenaQueueClient(args.queue, timeout_s=120.0) if args.queue else None
        self.pool = ProcessPoolExecutor(max_workers=args.pregate_workers)
        self.judge_pending: dict[str, dict] = {}
        self._prompt_tokens: dict[tuple[str, str, str], int] = {}

    # ------------------------------------------------------------ budgeting
    def token_budget(self, cfg: C.ProbeConfig, text: str) -> int:
        """How many NEW tokens fit behind this exact prompt in this model's
        context. Asked of the server's own tokenizer once per cell and cached.

        The previous probe lost an entire 16-candidate cell to `HTTPError 400`
        because a ~4.4k-token prompt plus a 12000-token request overflowed a
        16384 window, and the cell recorded 16 empty generations that read as
        model failures. A char/4 estimate is not good enough to bet a cell on.
        """
        spec = C.MODELS[cfg.model]
        key = (cfg.model, cfg.variant, cfg.task)
        n = self._prompt_tokens.get(key)
        if n is None:
            n = self.samplers[cfg.model].count_tokens(text)
            if n is None:
                n = len(text) // 3  # conservative fallback: overestimate
            self._prompt_tokens[key] = n
            print(f"[probe] {cfg.name}: prompt is {n} tokens", flush=True)
        room = spec.max_model_len - n - spec.reserve_tokens
        return max(256, min(self.args.max_new_tokens, spec.max_new_tokens, room))

    # ------------------------------------------------------------- lifecycle
    def log(self, rec: dict):
        with self.lock:
            self.records.append(rec)
            self.jsonl.write(json.dumps(rec, default=str) + "\n")

    def time_left(self) -> float:
        return self.deadline - time.time()

    # ------------------------------------------------------------ generation
    def generate_round(self, cfg: C.ProbeConfig, rnd: int) -> list[dict]:
        spec = C.MODELS[cfg.model]
        prompt = get_prompt(cfg.task, cfg.variant)
        text = R.render(spec.renderer, prompt)
        max_tok = self.token_budget(cfg, text)
        gens = self.samplers[cfg.model].sample_group(
            text,
            self.args.group_size,
            max_tokens=max_tok,
            temperature=self.args.temperature,
            stop=R.STOPS[spec.renderer],
        )
        seam = SEAMS.get(cfg.task) if cfg.is_seam else None
        recs = []
        for i, g in enumerate(gens):
            # A seam answer is a FILL, not a program: extract the named
            # definitions and compose them with the harness scaffold. What the
            # judge grades is `program`; what the MODEL wrote is `fill`. Both
            # are cached, because "the model got the algorithm right and the
            # harness dropped a block" and "the model wrote nothing usable"
            # are different findings and only the raw text separates them.
            if seam is not None:
                fill, how, missing = extract_fill(g["text"], seam.required)
                program = compose(cfg.task, fill) if fill.strip() else ""
            else:
                fill, how, missing = g["code"], g["extraction"], []
                program = g["code"]
            recs.append(
                {
                    "config": cfg.name,
                    "model": cfg.model,
                    "variant": cfg.variant,
                    "task": cfg.task,
                    "round": rnd,
                    "idx": i,
                    "group": f"{cfg.name}#r{rnd}",
                    "code_sha": sha(program),
                    "code_chars": len(program),
                    "fill_chars": len(fill),
                    "fill_approx_tokens": len(fill) // 4,
                    "gen_chars": len(g["text"]),
                    "max_new_tokens": max_tok,
                    "extraction": how,
                    "seam_missing_names": missing,
                    "finish_reason": g["finish_reason"],
                    "text": g["text"],
                    "fill": fill,
                    "code": program,
                }
            )
        return recs

    # -------------------------------------------------------------- pipeline
    def process_round(self, cfg: C.ProbeConfig, rnd: int):
        t_gen = time.time()
        recs = self.generate_round(cfg, rnd)
        gen_s = time.time() - t_gen
        print(f"[probe] {cfg.name} r{rnd}: {len(recs)} gens in {gen_s:.0f}s", flush=True)

        sigs = self.sigs[cfg.task]
        futs = {
            i: self.pool.submit(pregate_one, cfg.task, r["code"], sigs, timeout_s=self.args.pregate_timeout_s)
            for i, r in enumerate(recs)
        }
        for i, r in enumerate(recs):
            try:
                v = futs[i].result(timeout=self.args.pregate_timeout_s + 120)
            except Exception as e:
                v = {"passed": False, "gate": "harness", "reward": 0.0, "violations": [repr(e)]}
            r["pregate_gate"] = v.get("gate")
            r["pregate_passed"] = bool(v.get("passed"))
            r["pregate_observation"] = v.get("observation")
            r["gen_s"] = gen_s / max(1, len(recs))
            if not r["pregate_passed"]:
                r["gate"] = v.get("gate")
                r["reward"] = 0.0
                r["stage"] = "pregate"
                r["observation"] = v.get("observation")
                self.log(r)
        survivors = [r for r in recs if r["pregate_passed"]]
        n_pass = len(survivors)
        print(f"[probe] {cfg.name} r{rnd}: {n_pass}/{len(recs)} survived the pre-gate", flush=True)
        if not survivors:
            return
        if self.queue is None:
            for r in survivors:
                r.update(gate="pregate-only", reward=None, stage="pregate")
                self.log(r)
            return
        for r in survivors:
            wid = self.queue.submit(r["task"], r["code"], tag=f"{r['group']}#{r['idx']}")
            with self.lock:
                self.judge_pending[wid] = r

    # ------------------------------------------------------------- judge poll
    def drain_judge(self, block_until_empty: bool = False, timeout_s: float = 0.0):
        end = time.time() + timeout_s
        while True:
            with self.lock:
                pending = list(self.judge_pending)
            if not pending:
                return
            got = self.queue.poll_bulk(pending)
            for wid, rec in got.items():
                with self.lock:
                    r = self.judge_pending.pop(wid, None)
                if r is None:
                    continue
                res = rec.get("result") or {}
                r["stage"] = "judge"
                r["gate"] = res.get("gate")
                r["reward"] = res.get("reward", 0.0) or 0.0
                r["passed"] = bool(res.get("passed"))
                r["observation"] = res.get("observation")
                r["judge"] = {
                    k: res.get(k)
                    for k in (
                        "candidate_compile_s", "warm_chip_s", "item_wall_s", "export_s",
                        "peak_hbm_bytes", "latencies", "violations", "terminal", "score",
                    )
                }
                r["attempts"] = rec.get("attempts")
                self.log(r)
            if not block_until_empty:
                return
            if time.time() > end:
                with self.lock:
                    left = list(self.judge_pending.items())
                for wid, r in left:
                    r.update(stage="judge-timeout", gate="unjudged", reward=None)
                    self.log(r)
                    self.judge_pending.pop(wid, None)
                return
            time.sleep(5)

    # ------------------------------------------------------------------- run
    def main(self) -> int:
        if not self.configs:
            print("[probe] no runnable configurations", flush=True)
            return 1
        print(f"[probe] {len(self.configs)} configurations, {self.args.rounds} rounds of "
              f"{self.args.group_size}, wall {self.args.wall_s:.0f}s", flush=True)
        for rnd in range(self.args.rounds):
            if self.time_left() < self.args.round_reserve_s:
                print(f"[probe] stopping before round {rnd}: {self.time_left():.0f}s left", flush=True)
                break
            with ThreadPoolExecutor(max_workers=len(self.samplers)) as ex:
                # one thread per SERVER: two models generate concurrently,
                # each model's configurations serially so we never queue two
                # big prefills against the same engine.
                by_model: dict[str, list] = {}
                for cfg in self.configs:
                    by_model.setdefault(cfg.model, []).append(cfg)

                def run_model(cfgs):
                    for cfg in cfgs:
                        if self.time_left() < 60:
                            return
                        try:
                            self.process_round(cfg, rnd)
                        except Exception as e:
                            print(f"[probe] {cfg.name} r{rnd} FAILED: {e!r}", flush=True)

                list(ex.map(run_model, by_model.values()))
            if self.queue is not None:
                self.drain_judge()
            self.snapshot()
        # final drain
        if self.queue is not None:
            print("[probe] draining the judge queue", flush=True)
            self.drain_judge(block_until_empty=True, timeout_s=max(60.0, min(self.time_left(), 1800.0)))
        self.snapshot()
        return 0

    def noise_floors(self) -> dict:
        """task -> the judge's own measured noise floor, from its boot report.

        Without this the headline cannot be computed at all: a group's spread
        is only signal RELATIVE to the jitter of the timing protocol that
        produced it. Re-read every snapshot, because the boot report is
        fetched from the judge asynchronously.
        """
        p = self.args.boot_report
        if not p or not Path(p).exists():
            return {}
        try:
            rep = json.loads(Path(p).read_text())
        except Exception:
            return {}
        return {t: r.get("noise_floor") for t, r in rep.items() if isinstance(r, dict) and r.get("noise_floor")}

    def snapshot(self):
        from pallas_arena.probe.metrics import summarize

        with self.lock:
            recs = list(self.records)
        summary = summarize(recs, group_size=self.args.group_size, noise_floors=self.noise_floors())
        summary["wall_s"] = time.time() - self.t0
        self.out.write_text(json.dumps(summary, indent=1, default=str))
        head = summary.get("headline", {})
        print(
            f"[probe] {len(recs)} candidates | signal groups: {head.get('overall_signal_groups')} "
            f"(non-uniform: {head.get('overall_nonuniform_groups')})",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True, help="model=url,model=url")
    ap.add_argument("--queue", default=None, help="arena queue base url (omit for a pre-gate-only dry run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=13)  # 13 x 16 = 208 >= 200
    ap.add_argument("--group-size", type=int, default=C.GROUP_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--models", default=None)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--pregate-workers", type=int, default=14)
    ap.add_argument("--pregate-timeout-s", type=float, default=180.0)
    ap.add_argument("--wall-s", type=float, default=5400.0)
    ap.add_argument("--round-reserve-s", type=float, default=600.0)
    ap.add_argument(
        "--boot-report",
        default=None,
        help="judge boot-report json; supplies the per-task noise floor the headline is measured against",
    )
    args = ap.parse_args()
    return Probe(args).main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Muse-Glimmer reasoning-strength generation client (runs ON the TPU host).

Talks to the local vLLM OpenAI server with explicit ``list[int]`` prompts and
``return_token_ids``, so nothing round-trips through text and no tokenizer or
BOS discrepancy can masquerade as a model difference -- the same convention
``mg_client.py`` used for the parity work.

TWO-PHASE BUDGET, mirroring ``QwenTwoPhaseTokenCompleter`` in the discover
repo, transliterated into Muse-Glimmer's channel syntax:

  phase 1  prompt + reasoning, capped at ``phase1_max_tokens`` TOTAL
           (i.e. ``phase1_max_tokens - prompt_len`` new tokens).
  phase 2  only if phase 1 ran out of budget while still in ``to=self``:
           inject a masked close + answer cue

               <|eom|><|start|>assistant to=user<|message|>
               Here is the final complete program:\n\n```python\n

           and resample with whatever context is left. Without this a
           budget-exhausted rollout is simply an unparseable format error,
           which would show up as "xhigh is worse" when the truth is "xhigh
           needed more tokens".

Every raw generation (full reasoning + answer text, both phases, token counts)
is appended to a JSONL file BEFORE anything is graded. Regenerating is
expensive and a prior run in this project threw away 240 rollouts by saving
only metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

EOM = "<|eom|>"


def answer_cue(lang: str = "python") -> str:
    """The phase-2 forced answer opening.

    The fence language MUST match the environment's own
    ``_get_code_languages()``. `frontier_algo` (JSSP) is a C++17 problem, and
    cueing it with ```python made the model write Python for a C++ grader --
    320 rollouts scored "No C++ program with main() found in response".  The
    language now comes from the manifest, which records what the env asks for.
    """
    return (
        "<|eom|><|start|>assistant to=user<|message|>"
        f"Here is the final complete program:\n\n```{lang}\n"
    )


ANSWER_CUE = answer_cue()


def post(url: str, payload: dict, timeout: int = 3600) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # vLLM puts the actual reason in the BODY; a bare "HTTP 400" cost a
        # whole phase of slice time in the follow-up run.
        body = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {e.code}: {body}") from None


class Gauge:
    """Live in-flight counter.

    The whole reason the previous attempt ground for 19 hours is that nobody
    could see the achieved concurrency until the first 16 items finished --
    which, at ~11k tokens an item, took 1012 s.  This is checked every 20 s
    instead, so a client that is silently running 4-wide is visible inside two
    minutes.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = 0
        self.started = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.inflight += 1
            self.started += 1
            self.peak = max(self.peak, self.inflight)

    def leave(self):
        with self.lock:
            self.inflight -= 1


class Engine:
    def __init__(self, bases: list[str], model: str, tokenizer, gauge=None):
        self.bases = bases
        self.model = model
        self.tok = tokenizer
        self.gauge = gauge
        self._rr = 0
        self._rr_lock = threading.Lock()

    def _base(self) -> str:
        if len(self.bases) == 1:
            return self.bases[0]
        with self._rr_lock:
            b = self.bases[self._rr % len(self.bases)]
            self._rr += 1
        return b

    def complete(self, tokens: list[int], max_tokens: int, temperature: float,
                 stop_ids: list[int], ignore_eos: bool = False,
                 ) -> tuple[list[int], str]:
        payload = {
            "model": self.model,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_token_ids": stop_ids,
            "return_token_ids": True,
        }
        if ignore_eos:
            payload["ignore_eos"] = True
        if self.gauge:
            self.gauge.enter()
        try:
            r = post(f"{self._base()}/v1/completions", payload)
        finally:
            if self.gauge:
                self.gauge.leave()
        ch = r["choices"][0]
        ids = ch.get("token_ids") or r.get("token_ids") or []
        if not ids and ch.get("text"):
            ids = self.tok.encode(ch["text"], add_special_tokens=False)
        return list(ids), ch.get("finish_reason", "")


def split_channels(text: str) -> tuple[str, str, bool]:
    """(reasoning, answer, saw_reasoning_channel) from a raw completion.

    Kept deliberately simple and local so the TPU host needs no discover
    install; ``rosetta_stone.parse`` is the authority and the analysis step
    re-parses with it.
    """
    saw = "to=self" in text
    if "to=user" in text:
        head, _, tail = text.partition("to=user")
        ans = tail.split("<|message|>", 1)[1] if "<|message|>" in tail else tail
        think = ""
        if "to=self" in head:
            t = head.split("to=self", 1)[1]
            think = t.split("<|message|>", 1)[1] if "<|message|>" in t else t
        return think.replace(EOM, "").strip(), ans, saw
    if saw:
        t = text.split("to=self", 1)[1]
        think = t.split("<|message|>", 1)[1] if "<|message|>" in t else t
        return think.replace(EOM, "").strip(), "", saw
    return "", text, saw


def run_item(eng: Engine, item: dict, prompt: dict, args, cue_ids: list[int],
             stop_ids: list[int], cue_text: str = ANSWER_CUE) -> dict:
    t0 = time.time()
    ptoks = prompt["tokens"]
    plen = len(ptoks)
    budget = args.phase1_max_tokens - plen
    rec = {
        "item_id": item["item_id"],
        "prompt_key": item["prompt_key"],
        "rollout": item["rollout"],
        "arm": prompt["arm"],
        "kind": prompt["kind"],
        "state_id": prompt["state_id"],
        "run": prompt["run"],
        "parent_value": prompt["parent_value"],
        "prompt_len": plen,
        "phase1_budget": budget,
    }
    if budget <= 0:
        rec["error"] = f"prompt {plen} exceeds phase1_max_tokens {args.phase1_max_tokens}"
        return rec

    ids1, fin1 = eng.complete(ptoks, budget, args.temperature, stop_ids)
    txt1 = eng.tok.decode(ids1)
    rec.update(
        phase1_tokens=len(ids1),
        phase1_finish=fin1,
        phase1_truncated=len(ids1) >= budget,
        phase1_headroom=budget - len(ids1),
        text_phase1=txt1,
    )

    full = txt1
    rec["phase2_tokens"] = 0
    rec["phase2_used"] = False
    if len(ids1) >= budget and EOM not in txt1:
        # Budget exhausted mid-reasoning: force the channel switch and cue the
        # answer, exactly like the sweep's completer does for Qwen.
        answer_max = (
            args.context_window - plen - len(ids1) - len(cue_ids) - args.context_buffer
        )
        if answer_max > 0:
            ids2, fin2 = eng.complete(
                ptoks + ids1 + cue_ids, answer_max, args.temperature, stop_ids
            )
            txt2 = eng.tok.decode(ids2)
            full = txt1 + cue_text + txt2
            rec.update(
                phase2_tokens=len(ids2), phase2_finish=fin2, phase2_used=True,
                text_phase2=txt2, phase2_budget=answer_max,
            )
        else:
            rec["phase2_skipped"] = "no context left"

    think, answer, saw_self = split_channels(full)
    rec.update(
        text_full=full,
        reasoning=think,
        answer=answer,
        reasoning_tokens=len(eng.tok.encode(think, add_special_tokens=False)) if think else 0,
        answer_tokens=len(eng.tok.encode(answer, add_special_tokens=False)) if answer else 0,
        saw_reasoning_channel=saw_self,
        saw_answer_channel="to=user" in full,
        total_tokens=len(ids1) + rec["phase2_tokens"],
        wall_s=round(time.time() - t0, 2),
    )
    return rec


def run_probe(eng: Engine, man: dict, args, stop_ids: list[int],
              gauge: Gauge) -> float:
    """Fire ``--probe-conc`` concurrent requests on REAL prompts and measure
    aggregate tok/s.

    This exists because the previous attempt discovered its throughput only
    after 1012 s and 16 completed items.  Nothing here completes in two
    minutes at the real budget, so the only way to know the run is viable
    before committing hours to it is a short fixed-length burst at the exact
    concurrency the run will use.  It doubles as the XLA warm-up for that
    batch shape.
    """
    keys = [k for k in man["prompts"] if not k.startswith("smoke|")]
    keys.sort()
    if not keys:
        print("PROBE: no prompts", flush=True)
        return 0.0
    n = args.probe_conc
    picks = [man["prompts"][keys[i % len(keys)]] for i in range(n)]
    res = [0] * n
    errs = [""] * n

    def one(i):
        try:
            ids, _fin = eng.complete(picks[i]["tokens"], args.probe_tokens,
                                     args.temperature, stop_ids,
                                     ignore_eos=True)
            res[i] = len(ids)
        except Exception as e:  # noqa: BLE001
            errs[i] = repr(e)[:200]

    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    # heartbeat while the burst is in flight -- this is the concurrency proof
    while any(t.is_alive() for t in ths):
        time.sleep(10)
        print(f"  PROBE hb t={time.time()-t0:.0f}s inflight={gauge.inflight} "
              f"peak={gauge.peak}", flush=True)
    for t in ths:
        t.join()
    el = time.time() - t0
    tot = sum(res)
    ok = sum(1 for r in res if r)
    print(f"PROBE-RESULT conc={n} requested={n} ok={ok} "
          f"peak_inflight={gauge.peak} tokens={tot} wall={el:.1f}s "
          f"throughput={tot / max(el, 1e-9):.1f} tok/s "
          f"per_stream={tot / max(el, 1e-9) / max(ok, 1):.2f} tok/s", flush=True)
    bad = [e for e in errs if e]
    if bad:
        print(f"PROBE errors ({len(bad)}): {bad[0]}", flush=True)
    return tot / max(el, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8001",
                    help="comma-separated list; requests round-robin across them")
    ap.add_argument("--served-model", default="muse-glimmer-30b")
    ap.add_argument("--phase1-max-tokens", type=int, default=13824)
    ap.add_argument("--context-window", type=int, default=16384)
    ap.add_argument("--context-buffer", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heartbeat", type=int, default=20)
    ap.add_argument("--probe", action="store_true",
                    help="throughput/concurrency probe only, generate nothing")
    ap.add_argument("--probe-conc", type=int, default=64)
    ap.add_argument("--probe-tokens", type=int, default=512)
    ap.add_argument("--abort-after-errors", type=int, default=25)
    ap.add_argument("--answer-fence", default="",
                    help="override the phase-2 code fence language; "
                         "default comes from the manifest's code_languages")
    ap.add_argument("--deadline-epoch", type=int, default=0,
                    help="stop launching new items after this unix time")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True,
                                        local_files_only=True)
    gauge = Gauge()
    bases = [b.strip() for b in args.base.split(",") if b.strip()]
    eng = Engine(bases, args.served_model, tok, gauge)
    _man_head = json.load(open(args.manifest))
    lang = args.answer_fence or (_man_head.get("code_languages") or ["python"])[0]
    CUE = answer_cue(lang)
    cue_ids = tok.encode(CUE, add_special_tokens=False)
    print(f"answer-cue fence=```{lang}", flush=True)

    def single(s: str) -> int:
        # Same lookup the renderer uses (`encode` + assert length 1) rather
        # than convert_tokens_to_ids, whose behaviour varies by tokenizer
        # backend class -- and TokenizersBackend is new in transformers 5.x.
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"expected a single token for {s}, got {ids}"
        return ids[0]

    stop_ids = sorted({single("<|eot|>"), single("<|end_of_text|>")})
    print(f"stop_token_ids={stop_ids} cue_len={len(cue_ids)} bases={bases}",
          flush=True)

    man = _man_head
    if args.probe:
        run_probe(eng, man, args, stop_ids, gauge)
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --probe")
    items = man["items"]
    if args.limit:
        items = items[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # Only a SUCCESSFUL generation counts as done. If the engine
                # dies mid-run every remaining worker writes an error record in
                # seconds; treating those as done would let one crash poison
                # the whole manifest and make the resume a no-op.
                if "error" not in rec:
                    done.add(rec["item_id"])
    todo = [it for it in items if it["item_id"] not in done]
    print(f"items={len(items)} already_done={len(done)} todo={len(todo)}", flush=True)

    lock = threading.Lock()
    fh = open(args.out, "a", buffering=1)
    q: queue.Queue = queue.Queue()
    for it in todo:
        q.put(it)
    stats = {"ok": 0, "err": 0, "tok": 0, "t0": time.time(), "consec_err": 0}
    abort = threading.Event()

    def worker():
        while True:
            if abort.is_set():
                return
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            if args.deadline_epoch and time.time() > args.deadline_epoch:
                return
            try:
                rec = run_item(eng, it, man["prompts"][it["prompt_key"]], args,
                               cue_ids, stop_ids, CUE)
            except Exception as e:  # never lose the queue to one bad request
                rec = {"item_id": it["item_id"], "prompt_key": it["prompt_key"],
                       "rollout": it["rollout"], "error": repr(e)[:500]}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                if "error" in rec:
                    stats["err"] += 1
                    stats["consec_err"] += 1
                    # A dead engine turns 300 queued items into 300 error
                    # records in about a second. Stop instead, so the driver
                    # sees it and the queue survives for the next attempt.
                    if stats["consec_err"] >= args.abort_after_errors:
                        print(f"ABORT-ENGINE-DOWN after {stats['consec_err']} "
                              f"consecutive errors: {rec.get('error')}", flush=True)
                        abort.set()
                else:
                    stats["ok"] += 1
                    stats["consec_err"] = 0
                    stats["tok"] += rec.get("total_tokens", 0)
                n = stats["ok"] + stats["err"]
                if n % 8 == 0:
                    el = time.time() - stats["t0"]
                    print(
                        f"  {n}/{len(todo)} ok={stats['ok']} err={stats['err']} "
                        f"tok={stats['tok']} {stats['tok']/max(el,1):.0f} tok/s "
                        f"{el:.0f}s elapsed",
                        flush=True,
                    )

    nthreads = min(args.concurrency, max(1, len(todo)))
    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(nthreads)]
    stop_hb = threading.Event()

    def heartbeat():
        # ACHIEVED concurrency, printed from the first 20 s. `--concurrency N`
        # is a request; `inflight` is what actually happened.
        while not stop_hb.wait(args.heartbeat):
            el = time.time() - stats["t0"]
            print(
                f"  HB t={el:.0f}s inflight={gauge.inflight} peak={gauge.peak} "
                f"threads={nthreads} started={gauge.started} "
                f"done={stats['ok']} err={stats['err']} tok={stats['tok']} "
                f"{stats['tok']/max(el,1):.0f} tok/s(completed)",
                flush=True,
            )

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    print(f"LAUNCHING {nthreads} worker threads (requested {args.concurrency})",
          flush=True)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_hb.set()
    fh.close()
    el = time.time() - stats["t0"]
    print(
        f"DONE ok={stats['ok']} err={stats['err']} tokens={stats['tok']} "
        f"wall={el:.0f}s throughput={stats['tok']/max(el,1):.1f} tok/s "
        f"peak_inflight={gauge.peak}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())

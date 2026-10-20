"""SimpleTES search loop with our resume / budget / signal semantics.

``search`` is shared by every launcher; ``tpu.simpletes.local_driver`` runs it on della
as the child of the RQ4 local runner (``local_notrain --driver simpletes``): generation
on the run's leased inference farms through the local ingress, grading on the farms
through the same transport the RQ4 frozen client uses. Everything lives under the run's
``simpletes/`` folder; a restarted driver resumes from the newest complete checkpoint.

Spec keys: task (tpu.simpletes.tasks), num_chains, k_candidates, max_generations
(candidates, = RQ4 rollouts), eval_concurrency, selector="rpucg", num_inspirations,
reflection=true, log_interval, init_eval_repeats.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REFLECTION_ANSWER = 2048


def event(kind, **fields):
    print(json.dumps(dict(event=kind, time=time.time(), **fields)), flush=True)


def save_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()))
    tmp.replace(path)


def checkpoint_consistent(directory: Path) -> bool:
    """Complete and self-consistent: every node the policy's chains reference is in nodes.

    A GCS writeback can land mid-checkpoint, and SimpleTES names dirs by HHMMSS only, so
    a restored dir can mix files from two checkpoints; the cross-reference catches both.
    """
    try:
        if not all((directory / name).exists() for name in ("metadata.json", "config.json", "policy.json")):
            return False
        gz, plain = directory / "nodes.json.gz", directory / "nodes.json"
        if gz.exists():
            with gzip.open(gz, "rt") as stream:
                nodes = json.load(stream)
        elif plain.exists():
            nodes = json.loads(plain.read_text())
        else:
            return False
        ids = {node["id"] for node in nodes}
        policy = json.loads((directory / "policy.json").read_text())
        state = policy.get("state", policy)
        chains = [nid for chain in state.get("chain_history", {}).values() for nid in chain]
        return bool(ids) and all(nid in ids for nid in chains)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def newest_checkpoint(root: Path) -> str | None:
    """Most advanced consistent db_state_* dir; every other one is removed.

    SimpleTES sorts by HHMMSS name (wrong across midnight) and only deletes the dir it
    wrote itself, so restored stale dirs would be re-uploaded on every writeback.
    """
    ranked = []
    for directory in root.glob("*/instance-*/db_state_*"):
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
        except (OSError, ValueError):
            continue
        ranked.append(((metadata.get("completed_evaluations", 0), metadata.get("generation_attempts", 0),
                        directory.stat().st_mtime), directory))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = next((directory for _, directory in ranked if checkpoint_consistent(directory)), None)
    if chosen is not None:
        import shutil
        for directory in root.glob("*/instance-*/db_state_*"):
            if directory != chosen:
                shutil.rmtree(directory, ignore_errors=True)
    return str(chosen) if chosen else None


def guard_completion(engine) -> None:
    """Do not declare the run complete while a generation or evaluation is still in flight.

    Upstream ``_is_run_complete`` checks the generation *queue*, the eval queue and pending
    nodes, but a batch a generation worker has already taken is in none of them. Once the
    last batch is dispatched the run "completes" and cancels it mid-generation (seen live on
    Modal: 12 of 16 candidates generated, the final round of each chain lost).
    """
    original = engine._is_run_complete

    async def is_run_complete():
        if engine._gen_inflight or engine._eval_inflight:
            return False
        return await original()

    engine._is_run_complete = is_run_complete


def keep_previous_checkpoint(manager) -> None:
    """Delete the previous checkpoint only after the new one is written.

    Upstream removes the previous db_state_* before writing the next. A kill during that write
    (a round ends at a log_interval multiple, exactly when a step-boundary restart lands) left
    only a partial checkpoint, so the restart found nothing consistent and started over.
    """
    original = manager.snapshot

    async def snapshot(*args, **kwargs):
        previous, manager._last_checkpoint_path = manager._last_checkpoint_path, None
        await original(*args, **kwargs)
        if previous and previous != manager._last_checkpoint_path and os.path.isdir(previous):
            import shutil
            shutil.rmtree(previous, ignore_errors=True)

    manager.snapshot = snapshot


def reconcile_attempts(engine, k: int) -> tuple[int, int]:
    """Count only finalized batches after a resume.

    SimpleTES charges ``generation_attempts`` when a batch is *dispatched*, and a
    resume drops every in-flight batch (policy ``_batches`` is cleared) while the
    chains' ``chain_prompt_count`` only advances on finalize. Left alone, each
    preemption would silently spend up to C*K candidates that were never graded
    into a chain, and the run would stop short of the matched budget. Resetting the
    counter to finalized batches x K lets the lost batches be re-dispatched.
    """
    before = engine.generation_attempts
    engine.generation_attempts = sum(engine.selector.chain_prompt_count.values()) * k
    return before, engine.generation_attempts


def fit_prompts(engine, backend, max_prompt_tokens: int, log_path: Path) -> None:
    """Keep every generation prompt within the RQ4 caps by dropping the lowest-scored inspirations.

    RQ4 caps are relative to the prompt (thinking = phase1 - prompt), and a qubit program alone
    is ~4k tokens, so SimpleTES's five inspirations would leave no thinking budget. A prompt longer
    than ``max_prompt_tokens`` is rebuilt with its lowest-scored inspiration removed, down to one;
    every trim is logged (prompt_fit.jsonl) so results can disclose it.
    """
    from simpletes.node import score_key
    generator = engine.generator
    original = generator.build_prompt

    def build_prompt(inspirations, *args, **kwargs):
        kept = sorted(inspirations, key=score_key, reverse=True)
        while True:
            prompt = original(kept, *args, **kwargs)
            tokens = len(backend.prompt_ids(prompt))
            if tokens <= max_prompt_tokens or len(kept) <= 1:
                break
            kept = kept[:-1]
        if len(kept) < len(inspirations) or tokens > max_prompt_tokens:
            with log_path.open("a") as stream:
                stream.write(json.dumps(dict(time=time.time(), offered=len(inspirations), kept=len(kept),
                                             prompt_tokens=tokens, cap=max_prompt_tokens)) + "\n")
        return prompt

    generator.build_prompt = build_prompt


class TrimmedFailures(list):
    """SimpleTES failure records without the full prompt/output (tens of MB per checkpoint).

    Every rejected generation stores its whole prompt and output in failure.json, which
    is rewritten uncompressed at each checkpoint regardless of save_llm_io.
    """
    KEEP = 2000

    @classmethod
    def trim(cls, record):
        if isinstance(record, dict):
            record = dict(record)
            record.pop("llm_input", None)
            if isinstance(record.get("llm_output"), str):
                record["llm_output"] = record["llm_output"][-cls.KEEP:]
        return record

    def append(self, record):
        super().append(self.trim(record))


def token_totals(path: Path) -> dict:
    totals = dict(requests=0, candidates=0, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                  forced=0, no_code=0, bad_choices=0, overflows=0, reflection_requests=0,
                  reflection_tokens=0, reflection_overflows=0)
    if not path.exists():
        return totals
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == "generate":
            totals["requests"] += 1
            if row.get("overflow"):
                totals["overflows"] += 1
                continue
            for choice in row.get("choices", []):
                totals["candidates"] += 1
                totals["prompt_tokens"] += choice["prompt_tokens"]
                totals["completion_tokens"] += choice["completion_tokens"]
                totals["reasoning_tokens"] += choice.get("reasoning_tokens") or 0
                totals["forced"] += bool(choice.get("forced"))
                totals["no_code"] += bool(choice.get("no_code"))
                totals["bad_choices"] += "bad_choice" in choice
        elif row.get("kind") == "reflection":
            if row.get("overflow"):
                totals["reflection_overflows"] += 1
                continue
            totals["reflection_requests"] += 1
            totals["reflection_tokens"] += row.get("completion_tokens", 0)
    return totals


def prepare_run(folder: Path, spec: dict, task, model: str, member_spec: str):
    """Run identity (root construction, renderer date) that survives restarts."""
    State = task.env_type().state_type
    identity_path = folder / "run.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        root = State.from_dict(identity["root_state"])
    elif list((folder / "checkpoints").glob("*/instance-*/db_state_*")):
        # A partial restore: resuming these chains against a fresh random root would
        # silently mix two searches. Fail and let the job restart re-restore.
        raise RuntimeError(f"{identity_path} missing but checkpoints exist; refusing a new root")
    else:
        root = task.env_type().create_initial_state(task.problem_type)
        identity = dict(created=datetime.now(timezone.utc).isoformat(), task=task.name,
                        # Muse prompts carry the date: a run keeps the one it started with.
                        renderer_date=(spec.get("renderer_date") or os.environ.get("TTD_RENDERER_DATE")
                                       or datetime.now().strftime("%Y-%m-%d")),
                        root_state=root.to_dict(), spec=spec, model=model, member_spec=member_spec)
        save_json(identity_path, identity)
    return identity, root


def write_inputs(folder: Path, task, root, evaluator) -> None:
    """SimpleTES's instruction / init program / placeholder evaluator files."""
    from .tasks import init_program, science_instruction, science_starter
    if task.science:
        # RQ4 prompt without its starter; the starter becomes SimpleTES's init program.
        instruction, code = science_instruction(task), science_starter(task)
    else:
        instruction, code = evaluator._environment(root).get_question(), root.code
    (folder / "instruction.txt").write_text(instruction)
    (folder / "init_program.py").write_text(init_program(task, code))
    (folder / "evaluator.py").write_text("raise RuntimeError('graded by tpu.simpletes.evaluator, not a subprocess')\n")


async def search(folder: Path, spec: dict, task, evaluator, backend, caps, model: str, run_id: str,
                 seed: int = 0) -> int:
    """The SimpleTES loop with our resume / budget / signal semantics; shared by every launcher."""
    from .compat import import_simpletes
    import_simpletes()
    import simpletes.generator as generator_module
    from simpletes.config import EngineConfig
    from simpletes.engine import SimpleTESEngine

    num_chains, k = int(spec["num_chains"]), int(spec["k_candidates"])
    engine_config = EngineConfig(
        init_program=str(folder / "init_program.py"), evaluator_path=str(folder / "evaluator.py"),
        instruction_path=str(folder / "instruction.txt"),
        max_generations=int(spec["max_generations"]),
        num_inspirations=int(spec.get("num_inspirations", 5)),
        eval_concurrency=int(spec.get("eval_concurrency", num_chains * k)),
        eval_timeout=float(task.timeout() + 300),
        init_eval_repeats=int(spec.get("init_eval_repeats", 16)),
        gen_concurrency=num_chains, model=model, temperature=1.0,
        reflection_mode=bool(spec.get("reflection", True)),
        # save_llm_io off: every checkpoint rewrites all nodes and the executor re-uploads the
        # client dir each minute, so ~30k-token prompts per node would reach GBs. Code, metrics
        # and reflections are still saved; per-call tokens live in llm_calls.jsonl.
        output_path=str(folder / "checkpoints"), log_interval=int(spec.get("log_interval", 128)),
        save_llm_io=False, use_gzip=True, selector=spec.get("selector", "rpucg"),
        num_chains=num_chains, k_candidates=k, stream_k_candidates=False,
        include_construction=task.construction_global is not None, backpressure_multiplier=0)

    generator_module.create_llm_client = lambda _config: backend
    resume = newest_checkpoint(folder / "checkpoints")
    engine = SimpleTESEngine(engine_config, resume_path=resume)
    keep_previous_checkpoint(engine.checkpoint_manager)
    engine.worker = evaluator

    async def reflect(messages, temperature=0.7, max_tokens=None):
        # Same thinking cap as generation: a reflection that hits the cap gets the server's forced
        # "final complete program" cue, answers with code, and SimpleTES discards it (needs "Approach").
        # SimpleTES's own max_tokens (2048) bounds the answer after thinking.
        return await backend.generate_text(
            messages[-1]["content"], thinking_tokens=caps.thinking,
            answer_tokens=(max_tokens or REFLECTION_ANSWER) + 64,
            temperature=temperature, kind="reflection")

    engine.selector._llm_generate = reflect
    if spec.get("prompt") == "scaffold":
        from .scaffold import install
        install(engine, task, evaluator, evaluator.root_state, spec, folder, seed)
    # Prompts fit the budget: the fixed total, or at least min_thinking_tokens of RQ4 thinking (default 4096).
    fit_prompts(engine, backend, caps.max_prompt_tokens(int(spec.get("min_thinking_tokens", 4096))),
                folder / "prompt_fit.jsonl")
    engine._failure_records = TrimmedFailures()
    guard_completion(engine)
    if resume:
        engine.load_checkpoint(resume)
        engine._failure_records = TrimmedFailures(map(TrimmedFailures.trim, engine._failure_records))
        dispatched, finalized = reconcile_attempts(engine, k)
        record = dict(resumed_at=time.time(), checkpoint=resume, completed=engine.completed_evaluations,
                      dispatched=dispatched, finalized=finalized, redispatched=dispatched - finalized)
        with (folder / "resumes.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        event("simpletes_resumed", run_id=run_id, **record)
    event("simpletes_started", run_id=run_id, task=task.name, chains=num_chains, k=k,
          budget=engine_config.max_generations, context_window=caps.context_window,
          phase1_max_tokens=caps.phase1_max_tokens, fixed_thinking=caps.fixed_thinking,
          fixed_total=caps.fixed_total, num_inspirations=engine_config.num_inspirations,
          prompt=spec.get("prompt", "simpletes"), start=spec.get("start"), eager=spec.get("eager", 1.0))

    def hard_exit(sig):
        # SimpleTES's graceful stop finalizes every cancelled in-flight batch as failed, which
        # spends chain budget on ungraded candidates. Exit instead: the newest periodic checkpoint
        # plus reconcile_attempts re-dispatches everything that was in flight.
        event("simpletes_signal_exit", run_id=run_id, signal=int(sig))
        os._exit(128 + int(sig))

    async def take_over_signals():
        # Runs after engine.run() has installed its own handlers (first await), replacing them.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, hard_exit, sig)

    signals = asyncio.create_task(take_over_signals())
    try:
        await engine.run()
    finally:
        await signals
        await backend.aclose()

    best = engine.db.nodes.get(engine.best_node_id) if engine.best_node_id else None
    summary = dict(run_id=run_id, task=task.name, model=model,
                   num_chains=num_chains, k_candidates=k, max_generations=engine_config.max_generations,
                   generation_attempts=engine.generation_attempts,
                   completed_evaluations=engine.completed_evaluations,
                   # > max_generations only if a preemption orphaned part of a batch (see resumes.jsonl)
                   graded_nodes=len(engine.db.nodes) - 1,
                   generation_failures=engine.generation_failures,
                   evaluation_failures=engine.evaluation_failures,
                   best_combined_score=engine.best_score,
                   best_metric=(best.metrics or {}).get(task.metric) if best else None,
                   best_node_id=engine.best_node_id, checkpoint_dir=engine.checkpoint_dir,
                   tokens=token_totals(folder / "llm_calls.jsonl"))
    save_json(folder / "summary.json", summary)
    event("simpletes_complete", **summary)
    return 0 if engine.generation_attempts >= engine_config.max_generations else 1

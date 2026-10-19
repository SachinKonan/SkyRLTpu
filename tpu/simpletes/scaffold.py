"""SimpleTES arm of the scaffold experiment (agreed 2026-10-19).

Changes to upstream SimpleTES, each matching our TTT-Discover arm:

* prompt -- the problem's shared task text (word for word the text our arm shows before
  its parent block), then the inspirations, an optional warm-start line, the failure
  patterns, a blank line and SimpleTES's own closing line;
* start -- one root per chain. ``none``: a code-less root holding a random starting
  construction (our arm's empty parent), never shown as an inspiration. ``boot``:
  the 1024 drafts of the matching run of our arm, split in order into one group per
  chain; chain i starts from the best valid draft of group i;
* warm start -- the candidate is graded with the best inspiration's construction
  (Erdős ``initial_h_values``), as our arm grades with its parent's;
* eager commit -- a chain commits the best of its batch once ``eager`` of its K
  candidates are graded (our arm builds each step from the first 95% of samples);
  late candidates still enter the archive;
* snapshots -- ``steps/simpletes_step_NNNNNN.json`` when every chain has finished
  round N (one round = C x K samples = one step of our arm).
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np

from .tasks import EVOLVE_END, EVOLVE_START, strip_fences, strip_markers

GENERATION_INSTRUCTION = ("Generation instruction (must follow exactly):\n"
                          "1) Return your complete program in the answer format the task specifies, "
                          "as a single code block.\n\n")
CLOSING = ("Analyze the metrics above and create an improved program with higher combined_score. "
           "Try diverse approaches to solve the problem. Think outside the box. ")
WARM = {"erdos": "You may want to start your search from the best inspiration's construction, "
                 "which you can access through the `initial_h_values` global variable."}
WARM_EMPTY = {"erdos": "You may want to start your search from the current construction, "
                       "which you can access through the `initial_h_values` global variable."}
# Where our arm's prompt leaves the shared task text and starts the parent block.
TASK_END = {"erdos": "You are iteratively optimizing"}
PLACEHOLDER = f"{EVOLVE_START}\n# no program yet: empty starting point\n{EVOLVE_END}\n"
HIDDEN_METRICS = ("validity",)


def task_text(task, evaluator, root) -> str:
    if task.science:
        from .tasks import science_instruction
        return science_instruction(task).strip()
    question = evaluator._environment(root).get_question()
    return question[:question.index(TASK_END[task.name])].strip()


def is_placeholder(node) -> bool:
    return not node.parent_ids and node.code == PLACEHOLDER


class Arm:
    def __init__(self, engine, task, evaluator, root, spec: dict, folder: Path, seed: int):
        self.engine, self.task, self.evaluator, self.root = engine, task, evaluator, root
        self.spec, self.folder, self.seed = spec, folder, seed
        self.text = task_text(task, evaluator, root)
        self.root_ids: list[str] = []
        self.rounds_written = -1

    # ---- constructions -------------------------------------------------------------
    def construction_path(self, node_id: str) -> str:
        return os.path.join(self.engine._shared_construction_dir, f"{node_id}.json")

    def has_construction(self, node) -> bool:
        return bool(self.task.construction_global) and node is not None and os.path.exists(
            self.construction_path(node.id))

    # ---- prompt --------------------------------------------------------------------
    def build_prompt(self, inspirations, failure_patterns=None, policy_context="",
                     shared_construction_summary=None) -> str:
        from simpletes.generator import format_failure_patterns, format_inspiration
        from simpletes.node import score_key
        ranked = sorted(inspirations, key=score_key, reverse=True)
        shown = [node for node in ranked if not is_placeholder(node)]
        parts = [f"Task: {self.text}\n\n", GENERATION_INSTRUCTION]
        if shown:
            parts.append("In-context inspirations (sorted by score, higher is better):\n")
            for index, node in enumerate(shown, 1):
                metrics = {k: v for k, v in (node.metrics or {}).items() if k not in HIDDEN_METRICS}
                parts.append(format_inspiration(index, strip_markers(node.code).strip("\n"), node.score,
                                                metrics, node.reflection, "python"))
        if ranked and self.has_construction(ranked[0]):
            parts.append(f"\n{(WARM if shown else WARM_EMPTY)[self.task.name]}\n")
        if failure_patterns:
            parts.append(format_failure_patterns(failure_patterns))
        parts.append("\n" + CLOSING)
        return "".join(parts)

    # ---- roots ---------------------------------------------------------------------
    def root_nodes(self) -> list[tuple[object, list[float] | None]]:
        """(node, construction) per chain."""
        from simpletes.node import Node, Status
        chains = int(self.spec["num_chains"])
        start = self.spec.get("start", "none")
        roots = []
        if start == "none":
            env_type = self.task.env_type()
            for chain in range(chains):
                state = env_type.create_initial_state(self.task.problem_type, seed=self.seed * 1000 + chain)
                roots.append((float(state.value), PLACEHOLDER, state.construction))
        elif start == "boot":
            roots = [(value, f"{EVOLVE_START}\n{strip_fences(code).strip()}\n{EVOLVE_END}\n", construction)
                     for value, code, construction in bootstrap_roots(Path(self.spec["bootstrap_from"]), chains)]
        else:
            raise ValueError(f"unknown start {start!r}")
        nodes = []
        for value, code, construction in roots:
            metrics = {"combined_score": value}
            if self.task.minimize:
                metrics[self.task.metric] = -value
            node = Node(id=uuid.uuid4().hex, code=code, parent_ids=[], status=Status.DONE,
                        metrics=metrics, score=value)
            nodes.append((node, construction))
        return nodes

    async def initialize(self):
        """Replaces SimpleTES's init-program evaluation: one root per chain."""
        from simpletes.construction import encode_construction
        from simpletes.node import validate_node_for_db
        engine = self.engine
        roots = self.root_nodes()
        async with engine._db_lock:
            for chain, (node, construction) in enumerate(roots):
                validate_node_for_db(node)
                engine.db.add(node)
                self.root_ids.append(node.id)
                if node.score > engine.best_score:
                    engine.best_score, engine.best_node_id = node.score, node.id
                engine._chain_best_scores[chain] = node.score
                if construction is not None and self.task.construction_global:
                    engine._set_shared_construction_for_chain_locked(
                        chain, snapshot_id=node.id,
                        payload=encode_construction(np.asarray(construction, dtype=float)))
        self.write_round(0)

    def initialize_root(self, db) -> bool:
        selector = self.engine.selector
        if selector._initialized:
            return True
        if len(self.root_ids) != selector.num_chains:
            return False
        for chain, node_id in enumerate(self.root_ids):
            selector.chains[chain].append(node_id)
            selector.chain_history[chain].append(node_id)
            selector._insert_sorted(chain, node_id, db.nodes[node_id].score)
        selector._root_id = self.root_ids[0]
        selector._initialized = True
        return True

    # ---- warm start ----------------------------------------------------------------
    async def select_and_prompt(self, original):
        from simpletes.node import score_key
        prepared = await original()
        if prepared is None:
            return None
        prompt, inspiration_ids, chain, _ = prepared
        async with self.engine._db_lock:
            nodes = [self.engine.db.nodes[i] for i in inspiration_ids if i in self.engine.db.nodes]
        best = max(nodes, key=score_key) if nodes else None
        return prompt, inspiration_ids, chain, best.id if self.has_construction(best) else None

    async def commit(self, original, node, captured_construction_payload=None):
        if captured_construction_payload is not None and self.task.construction_global:
            from simpletes.construction import write_payload
            path = self.construction_path(node.id)
            if not os.path.exists(path):
                write_payload(path, captured_construction_payload)
        await original(node, captured_construction_payload)

    # ---- per-round snapshots -------------------------------------------------------
    def write_round(self, completed: int):
        if completed <= self.rounds_written:
            return
        engine, selector = self.engine, self.engine.selector
        db = engine.db.nodes
        best = db.get(engine.best_node_id) if engine.best_node_id else None

        def state(node, chain=None):
            return dict(id=node.id, chain=chain if chain is not None else node.chain_idx, value=node.score,
                        metrics=node.metrics, parents=list(node.parent_ids), reflection=node.reflection,
                        placeholder=is_placeholder(node), code=strip_markers(node.code))

        committed = [state(db[nid], chain) for chain, ids in selector.chain_history.items()
                     for nid in ids if nid in db]
        out = self.folder / "steps"
        out.mkdir(exist_ok=True)
        path = out / f"simpletes_step_{completed:06d}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(step=completed, time=time.time(), evaluated=len(db),
                                       generation_attempts=engine.generation_attempts,
                                       best=state(best) if best else None, states=committed), default=float))
        tmp.replace(path)
        self.rounds_written = completed

    async def after_finalize(self, original, completion):
        await original(completion)
        counts = self.engine.selector.chain_prompt_count
        if counts:
            self.write_round(min(counts.values()))


def bootstrap_roots(bootstrap: Path, chains: int) -> list[tuple[float, str, list[float] | None]]:
    """Best valid draft of each of ``chains`` equal, in-order groups of our arm's bootstrap drafts."""
    if not (bootstrap / "complete.json").exists():
        raise RuntimeError(f"{bootstrap} has no complete.json: run our arm's bootstrap first")
    drafts = []
    for group in sorted((bootstrap / "drafts").glob("group-*")):
        for grade in sorted(group.glob("grade-*.json")):
            drafts.append(json.loads(grade.read_text()))
    if not drafts or len(drafts) % chains:
        raise RuntimeError(f"{len(drafts)} bootstrap drafts do not split into {chains} chains")
    size = len(drafts) // chains
    roots = []
    for chain in range(chains):
        best = None
        for draft in drafts[chain * size:(chain + 1) * size]:
            state = draft.get("state")
            if isinstance(state, str):
                state = json.loads(state)
            if float(draft.get("correctness") or 0) != 1.0 or not state or state.get("value") is None:
                continue
            if best is None or float(state["value"]) > float(best[0]):
                best = (float(state["value"]), state.get("code") or draft["code"], state.get("construction"))
        if best is None:
            raise RuntimeError(f"bootstrap group {chain} ({size} drafts) has no valid draft")
        roots.append(best)
    return roots


def eager(selector, fraction: float):
    """Finalize a batch once ceil(fraction x K) of its candidates are done; ignore the rest."""
    original_register = selector.register_batch

    def register_batch(gen_id, chain_idx, inspiration_ids, k):
        original_register(gen_id, chain_idx, inspiration_ids, k)
        with selector._lock:
            selector._batches[gen_id]["submitted"] = max(1, math.ceil(fraction * k))

    def once(original):
        # Calls run on the engine's event loop thread, so check-and-mark cannot interleave.
        def call(*args, **kwargs):
            pending = original(*args, **kwargs)
            if pending is None:
                return None
            batch = selector._batches.get(pending.gen_id)
            if batch is None or batch.get("finalizing"):
                return None
            batch["finalizing"] = True
            return pending
        return call

    selector.register_batch = register_batch
    selector.on_child_done = once(selector.on_child_done)
    selector.on_generation_failed = once(selector.on_generation_failed)


def install(engine, task, evaluator, root, spec: dict, folder: Path, seed: int) -> Arm:
    arm = Arm(engine, task, evaluator, root, spec, folder, seed)
    engine.generator.build_prompt = arm.build_prompt
    engine._initialize_from_scratch = arm.initialize
    engine.selector._initialize_root = arm.initialize_root
    select, commit, finalize = (engine._select_inspirations_and_prompt, engine._commit_node,
                                engine._handle_pending_finalize)
    engine._select_inspirations_and_prompt = lambda: arm.select_and_prompt(select)
    engine._commit_node = lambda node, payload=None: arm.commit(commit, node, payload)
    engine._handle_pending_finalize = lambda completion: arm.after_finalize(finalize, completion)
    if float(spec.get("eager", 1.0)) < 1.0:
        eager(engine.selector, float(spec["eager"]))
    return arm

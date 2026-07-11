"""Create a compact, replay-free Tinker database for SFT-to-RL handoff."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def prepare_seed(source: Path, output: Path, model_id: str, checkpoint_ids: list[str]) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("source and output databases must be different files")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    source_db = sqlite3.connect(source)
    output_db = sqlite3.connect(output)
    with output_db:
        source_db.backup(output_db)
    source_db.close()

    model = output_db.execute(
        "SELECT session_id, request_id FROM models WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    if model is None:
        raise ValueError(f"model {model_id!r} is absent from {source}")
    session_id, create_request_id = model
    if not checkpoint_ids:
        raise ValueError("at least one checkpoint ID is required")
    for checkpoint_id in checkpoint_ids:
        checkpoint = output_db.execute(
            "SELECT status FROM checkpoints WHERE model_id = ? AND checkpoint_id = ? AND checkpoint_type = 'TRAINING'",
            (model_id, checkpoint_id),
        ).fetchone()
        if checkpoint != ("COMPLETED",):
            raise ValueError(
                f"training checkpoint {model_id}/{checkpoint_id} is not completed: {checkpoint!r}"
            )

    output_db.execute("PRAGMA foreign_keys=OFF")
    output_db.execute("DELETE FROM sampling_sessions")
    placeholders = ",".join("?" for _ in checkpoint_ids)
    output_db.execute(
        f"DELETE FROM checkpoints WHERE model_id != ? OR checkpoint_id NOT IN ({placeholders}) "
        "OR checkpoint_type != 'TRAINING'",
        (model_id, *checkpoint_ids),
    )
    output_db.execute("DELETE FROM models WHERE model_id != ?", (model_id,))
    output_db.execute("DELETE FROM sessions WHERE session_id != ?", (session_id,))
    output_db.execute("DELETE FROM futures WHERE request_id != ?", (create_request_id,))
    output_db.execute("UPDATE models SET status = 'unloaded' WHERE model_id = ?", (model_id,))
    output_db.execute("UPDATE sessions SET status = 'expired' WHERE session_id = ?", (session_id,))
    output_db.commit()
    output_db.execute("VACUUM")

    pending = output_db.execute("SELECT COUNT(*) FROM futures WHERE status = 'PENDING'").fetchone()[0]
    retained = output_db.execute(
        "SELECT model_id, checkpoint_id, checkpoint_type, status FROM checkpoints"
    ).fetchall()
    expected = sorted((model_id, checkpoint_id, "TRAINING", "COMPLETED") for checkpoint_id in checkpoint_ids)
    if pending or sorted(retained) != expected:
        raise RuntimeError(f"invalid RL seed: pending={pending}, checkpoints={retained!r}")
    output_db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-id", action="append", dest="checkpoint_ids", required=True)
    args = parser.parse_args()
    prepare_seed(args.source, args.output, args.model_id, args.checkpoint_ids)
    print(f"prepared {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

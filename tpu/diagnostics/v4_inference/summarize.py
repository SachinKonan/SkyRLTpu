"""Produce a PID-correlated failure inventory from saved host snapshots."""
import argparse
import json
from pathlib import Path


def summarize(documents):
    rows = []
    for doc in documents:
        for run, data in doc.get("runs", {}).items():
            for engine in data["engines"]:
                rows.append(dict(host=doc["host"], run=run, engine=Path(engine["file"]).name,
                                 pids=engine["pids"], classification=engine["classification"],
                                 driver_files=list(engine["driver_errors"]),
                                 serving_versions=doc["environments"].get("serving", {})))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize([json.loads(p.read_text()) for p in args.snapshots]), indent=2))


if __name__ == "__main__":
    main()

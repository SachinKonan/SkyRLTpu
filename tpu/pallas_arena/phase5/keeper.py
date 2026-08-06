"""Fleet keeper: hold the arena judge fleet at N, and NEVER above it.

Spot judges get preempted; a stateless puller coming back is a non-event by
construction (the lease expires, the item regrades elsewhere), so the only
thing that has to be reconciled is the number of QRs. This is the phase-2
provisioning script wrapped in the forever_sweep reconciliation shape.

The invariant that matters more than availability: **never more than
`--size` arena QRs alive at once, and delete before create.** A keeper that
races (create the replacement, then delete the corpse) is exactly how a
quota-limited shared zone gets wedged for everyone. Every create in here is
preceded by a delete of the slot it is replacing and gated on a fresh count
of live QRs, and the total number of replacements is capped.

Slots are fixed names (`<prefix>-1..N`) so a stray QR from another run or
another user can never be mistaken for ours, and so `delete` is always
addressed to a name we chose.

  python -m pallas_arena.phase5.keeper --prefix sk7524-pallas-judge --size 5 \
      --zone us-east5-b --once            # reconcile once and print JSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

# States that still consume a QR slot (and quota) -> count as alive.
ALIVE = {"ACTIVE", "ACCEPTED", "CREATING", "PROVISIONING", "WAITING_FOR_RESOURCES"}
# States that will never serve work -> delete and free the slot.
DEAD = {"SUSPENDED", "SUSPENDING", "FAILED", "DELETING"}


def _gc(args_: list[str], zone: str, project: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gcloud", "compute", "tpus", *args_, f"--zone={zone}", f"--project={project}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def list_slots(prefix: str, size: int, zone: str, project: str) -> dict[str, str]:
    """slot name -> state ('' == does not exist). Only ever reports the
    fixed slot names, so nothing else in the zone can be touched."""
    r = _gc(["queued-resources", "list", "--format=value(name,state.state)"], zone, project)
    found = {}
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            found[parts[0]] = parts[1]
    return {f"{prefix}-{i}": found.get(f"{prefix}-{i}", "") for i in range(1, size + 1)}


def delete_slot(name: str, zone: str, project: str) -> int:
    return _gc(["queued-resources", "delete", name, "--force", "--quiet"], zone, project).returncode


def create_slot(name: str, zone: str, project: str, accel: str, runtime: str) -> int:
    return _gc(
        [
            "queued-resources",
            "create",
            name,
            f"--node-id={name}",
            f"--accelerator-type={accel}",
            f"--runtime-version={runtime}",
            "--spot",
        ],
        zone,
        project,
    ).returncode


def reconcile(args, budget: dict) -> dict:
    slots = list_slots(args.prefix, args.size, args.zone, args.project)
    alive = {n: s for n, s in slots.items() if s in ALIVE}
    dead = {n: s for n, s in slots.items() if s in DEAD}
    missing = [n for n, s in slots.items() if s == ""]
    actions: list[dict] = []

    # 1. delete first, always. Freeing the slot before asking for another is
    #    what keeps the hard ceiling a ceiling.
    for name, state in dead.items():
        rc = delete_slot(name, args.zone, args.project)
        actions.append({"action": "delete", "name": name, "state": state, "rc": rc})
        missing.append(name)

    # 2. create only into a slot we just proved is free, only while under the
    #    ceiling, and only up to the replacement cap.
    if args.replace:
        for name in missing:
            live_now = len(alive) + sum(1 for a in actions if a["action"] == "create" and a["rc"] == 0)
            if live_now >= args.size:
                actions.append({"action": "skip-create", "name": name, "why": f"at ceiling {args.size}"})
                continue
            if budget["creates"] >= args.max_replacements:
                actions.append({"action": "skip-create", "name": name, "why": "replacement budget spent"})
                continue
            budget["creates"] += 1
            rc = create_slot(name, args.zone, args.project, args.accel, args.runtime)
            actions.append({"action": "create", "name": name, "rc": rc, "n": budget["creates"]})

    return {
        "t": time.strftime("%H:%M:%S"),
        "slots": slots,
        "alive": sorted(alive),
        "active": sorted(n for n, s in slots.items() if s == "ACTIVE"),
        "dead": dead,
        "actions": actions,
        "replacements_used": budget["creates"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="sk7524-pallas-judge")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--zone", default="us-east5-b")
    ap.add_argument("--project", default="vision-mix")
    ap.add_argument("--accel", default="v6e-1")
    ap.add_argument("--runtime", default="v2-alpha-tpuv6e")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--max-replacements", type=int, default=3)
    ap.add_argument("--replace", action="store_true", help="recreate preempted slots (delete-before-create)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--deadline-s", type=float, default=0.0, help="stop reconciling after this long")
    args = ap.parse_args()

    budget = {"creates": 0}
    stop_at = time.time() + args.deadline_s if args.deadline_s else None
    while True:
        try:
            state = reconcile(args, budget)
        except Exception as e:
            state = {"t": time.strftime("%H:%M:%S"), "error": repr(e)}
        print(json.dumps(state, default=str), flush=True)
        if args.state_file:
            try:
                from pathlib import Path

                Path(args.state_file).write_text(json.dumps(state, indent=1, default=str))
            except Exception:
                pass
        if args.once or (stop_at and time.time() > stop_at):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

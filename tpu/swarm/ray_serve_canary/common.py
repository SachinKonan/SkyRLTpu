"""Small, dependency-free contracts for the direct-upload canary."""
import asyncio
import json
import math
import re


def adapter_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("invalid adapter version")
    return value


class VersionGate:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.updating = False
        self.active = 0
        self.version = None

    async def enter(self, version):
        async with self.condition:
            if self.updating:
                raise ValueError("adapter update in progress")
            if self.version is None or version != self.version:
                raise ValueError("requested adapter is not committed")
            self.active += 1

    async def leave(self):
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()

    async def begin_update(self):
        async with self.condition:
            self.updating = True
            await self.condition.wait_for(lambda: self.active == 0)

    async def commit(self, version):
        async with self.condition:
            self.version = version
            self.updating = False
            self.condition.notify_all()


def grade_construction(text):
    """Score generated data without executing model-generated Python."""
    decoder = json.JSONDecoder()
    result = None
    for i, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict) and "h_values" in value:
                    result = value
            except ValueError:
                pass
    if result is None:
        return {"valid": False, "reason": "no construction JSON", "reward": 0.0}
    h = result["h_values"]
    if not isinstance(h, list) or not 4 <= len(h) <= 1024:
        return {"valid": False, "reason": "invalid length", "reward": 0.0}
    if any(type(x) not in (float, int) or not 0 <= x <= 1 or not math.isfinite(x) for x in h):
        return {"valid": False, "reason": "invalid density", "reward": 0.0}
    total = sum(h)
    if total <= 0:
        return {"valid": False, "reason": "zero mass", "reward": 0.0}
    h = [x * len(h) / (2 * total) for x in h]
    if max(h) > 1:
        return {"valid": False, "reason": "normalized density exceeds one", "reward": 0.0}
    n = len(h)
    c5 = max(sum(h[i] * (1 - h[i - k]) for i in range(n) if 0 <= i - k < n)
             for k in range(1 - n, n)) * 2 / n
    return {"valid": True, "c5": c5, "reward": 1 / (1e-8 + c5), "n_points": n}

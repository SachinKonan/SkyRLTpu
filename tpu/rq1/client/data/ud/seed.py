"""Seed: axis-aligned 256x256 unit square lattice (each interior point pairs with
its right and up neighbor at distance exactly 1)."""


def run_construction():
    pts = []
    for i in range(256):
        for j in range(256):
            pts.append((float(i), float(j)))
    return pts

"""Fixed-trial configuration. Candidate may inspect, but should not modify evaluator settings."""
from __future__ import annotations
import json
from pathlib import Path

DEFAULTS = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def read_config(path=None, **overrides):
    with open(path or DEFAULTS, encoding="utf8") as f:
        config = json.load(f)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def noise_sigma(config):
    # Public Zenodo release: sigma_e / sqrt(2 * n_g * A_pix); A_pix = pixel_size^2.
    return float(config["noise_sigma_e"] / (
        2 * config["noise_ng"] * config["noise_pixel_arcmin"] ** 2
    ) ** 0.5)

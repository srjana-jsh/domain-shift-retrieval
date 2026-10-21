from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

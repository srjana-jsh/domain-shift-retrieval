from __future__ import annotations

import json
from pathlib import Path


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print()
    print(title)

    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")


def save_results(
    project_root: Path,
    method: str,
    dataset_name: str,
    top_k: int,
    metrics: dict[str, float],
    extra: dict | None = None,
) -> Path:
    """Save experiment metrics as a small JSON file."""

    output_dir = project_root / "reports" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_dataset_name = dataset_name.replace("/", "_")
    output_path = output_dir / f"{method}_{safe_dataset_name}.json"

    payload = {
        "method": method,
        "dataset": dataset_name,
        "top_k": top_k,
        "metrics": metrics,
    }

    if extra:
        payload.update(extra)

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return output_path
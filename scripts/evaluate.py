from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config
from data.datasets import load_retrieval_dataset
from evaluation.metrics import evaluate_retrieval
from retrieval.factory import build_retriever
from utils.results import print_metrics, save_result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval model.")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    dataset_name = config["dataset"]["name"]
    model_type = config["retrieval"]["model"]
    top_k = config["retrieval"]["top_k"]

    print(f"Loading dataset: {dataset_name}")

    dataset = load_retrieval_dataset(dataset_name)

    print(f"Documents: {len(dataset.documents):,}")
    print(f"Queries:   {len(dataset.queries):,}")

    retriever = build_retriever(config, dataset.documents)

    results = retriever.retrieve_all(
        queries=dataset.queries,
        top_k=top_k,
    )

    metrics = evaluate_retrieval(results, dataset.qrels)

    print_metrics(title=f"{model_type} results", metrics=metrics)

    output_path = save_results(
        project_root=PROJECT_ROOT,
        method=model_type,
        dataset_name=dataset_name,
        top_k=top_k,
        metrics=metrics,
    )

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
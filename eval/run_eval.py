"""Eval harness (STUB — implemented by Task 10).

Computes recall@1/5/10, top-1, per-stage latency, weak-match rate, index
freshness, and threshold calibration over a labeled dataset.

Usage: python eval/run_eval.py --dataset eval/dataset
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-search eval harness (stub).")
    parser.add_argument("--dataset", default="eval/dataset", help="Path to labeled eval dataset.")
    parser.parse_args()
    raise NotImplementedError("run_eval is implemented by Task 10.")


if __name__ == "__main__":
    main()

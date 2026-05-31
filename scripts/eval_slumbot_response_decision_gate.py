#!/usr/bin/env python3
"""Evaluate a learned selector for response-conditioned Slumbot resolving."""

from __future__ import annotations

import argparse
import json

from poker_ai.research.slumbot_response_decision_gate import (
    evaluate_response_decision_gate,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a response-range decision gate on one EV replay artifact and evaluate on another."
    )
    parser.add_argument("--train-ev-gate", required=True)
    parser.add_argument("--eval-ev-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    metrics = evaluate_response_decision_gate(
        args.train_ev_gate,
        args.eval_ev_gate,
        ridge_l2=args.ridge_l2,
        threshold=args.threshold,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

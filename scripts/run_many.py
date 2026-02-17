#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.game import parse_resolved_config
from src.io import deep_merge, load_yaml
from src.orchestrator import run_experiment


def _parse_models(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple independent game simulations and aggregate outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num-runs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--random-assign", action="store_true")
    args = parser.parse_args()

    base = load_yaml(Path(args.config))
    overrides: dict = {
        "runtime": {"num_runs": args.num_runs, "seed": args.seed},
        "game": {},
        "llm": {},
    }

    if args.rounds is not None:
        overrides["game"]["rounds"] = args.rounds
    if args.agents is not None:
        overrides["game"]["agents"] = args.agents
    if args.models is not None:
        overrides["llm"]["models"] = _parse_models(args.models)
    if args.random_assign:
        overrides["llm"]["random_assign"] = True

    merged = deep_merge(base, overrides)
    config = parse_resolved_config(merged)

    result = run_experiment(config=config, num_runs=config.runtime.num_runs, base_seed=config.runtime.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

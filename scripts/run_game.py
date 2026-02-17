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
from src.orchestrator import run_experiment, run_single_game


def _parse_model_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated public-goods game with HPC LLM agents.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")

    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--initial-wealth", type=float, default=None)
    parser.add_argument("--multiplier", type=float, default=None)

    parser.add_argument("--default-model", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model list.")
    parser.add_argument("--random-assign", action="store_true")
    parser.add_argument("--summarizer-model", default=None)
    parser.add_argument("--max-job-retries", type=int, default=None)
    parser.add_argument("--parse-repair-retries", type=int, default=None)

    parser.add_argument("--partition", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--cpus-per-gpu", type=int, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--time-limit", default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--poll-interval", type=int, default=None)

    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    base = load_yaml(Path(args.config))

    overrides: dict = {"game": {}, "llm": {}, "slurm": {}, "runtime": {}}

    if args.rounds is not None:
        overrides["game"]["rounds"] = args.rounds
    if args.agents is not None:
        overrides["game"]["agents"] = args.agents
    if args.initial_wealth is not None:
        overrides["game"]["initial_wealth"] = args.initial_wealth
    if args.multiplier is not None:
        overrides["game"]["multiplier"] = args.multiplier

    if args.default_model is not None:
        overrides["llm"]["default_model"] = args.default_model
    if args.models is not None:
        overrides["llm"]["models"] = _parse_model_list(args.models)
    if args.random_assign:
        overrides["llm"]["random_assign"] = True
    if args.summarizer_model is not None:
        overrides["llm"]["summarizer_model"] = args.summarizer_model
    if args.max_job_retries is not None:
        overrides["llm"]["max_job_retries"] = args.max_job_retries
    if args.parse_repair_retries is not None:
        overrides["llm"]["parse_repair_retries"] = args.parse_repair_retries

    if args.partition is not None:
        overrides["slurm"]["partition"] = args.partition
    if args.gres is not None:
        overrides["slurm"]["gres"] = args.gres
    if args.cpus_per_gpu is not None:
        overrides["slurm"]["cpus_per_gpu"] = args.cpus_per_gpu
    if args.mem is not None:
        overrides["slurm"]["mem"] = args.mem
    if args.time_limit is not None:
        overrides["slurm"]["time_limit"] = args.time_limit
    if args.conda_env is not None:
        overrides["slurm"]["conda_env"] = args.conda_env
    if args.poll_interval is not None:
        overrides["slurm"]["poll_interval"] = args.poll_interval

    if args.data_root is not None:
        overrides["runtime"]["data_root"] = args.data_root
    if args.num_runs is not None:
        overrides["runtime"]["num_runs"] = args.num_runs
    if args.seed is not None:
        overrides["runtime"]["seed"] = args.seed

    merged = deep_merge(base, overrides)
    config = parse_resolved_config(merged)

    num_runs = config.runtime.num_runs
    seed = config.runtime.seed

    if num_runs <= 1:
        result = run_single_game(config=config, run_index=0, run_seed=seed)
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "run_dir": str(result.run_dir),
                    "status": result.status,
                    "error": result.error,
                },
                indent=2,
            )
        )
        if result.status == "interrupted":
            raise SystemExit(130)
        if result.status != "completed":
            raise SystemExit(1)
    else:
        result = run_experiment(config=config, num_runs=num_runs, base_seed=seed)
        print(json.dumps(result, indent=2))
        if bool(result.get("interrupted")):
            raise SystemExit(130)


if __name__ == "__main__":
    main()

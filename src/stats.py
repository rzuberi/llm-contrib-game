
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.io import read_json, read_jsonl


def gini(values: List[float]) -> float:
    if not values:
        return 0.0

    clean = [max(0.0, float(v)) for v in values]
    total = sum(clean)
    if total <= 0:
        return 0.0

    sorted_vals = sorted(clean)
    n = len(sorted_vals)
    weighted_sum = 0.0
    for idx, value in enumerate(sorted_vals, start=1):
        weighted_sum += idx * value

    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def _rank_map(values: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    return {agent_id: idx + 1 for idx, (agent_id, _) in enumerate(ordered)}


def compute_round_stats(
    *,
    contributions: Dict[str, float],
    wealth_before: Dict[str, float],
    wealth_after: Dict[str, float],
    pot: float,
    redistributed_per_agent: float,
    contribution_history: List[Dict[str, float]],
    rolling_window: int,
) -> Dict[str, Any]:
    contrib_values = [float(v) for v in contributions.values()]
    wealth_values = [float(v) for v in wealth_after.values()]

    mean_contrib = statistics.fmean(contrib_values) if contrib_values else 0.0
    median_contrib = statistics.median(contrib_values) if contrib_values else 0.0
    mean_wealth = statistics.fmean(wealth_values) if wealth_values else 0.0
    std_wealth = statistics.pstdev(wealth_values) if len(wealth_values) > 1 else 0.0

    before_rank = _rank_map(wealth_before)
    after_rank = _rank_map(wealth_after)
    rank_change = {agent_id: before_rank[agent_id] - after_rank[agent_id] for agent_id in wealth_after}

    recent = contribution_history[-rolling_window:] if rolling_window > 0 else contribution_history
    rolling_avg: Dict[str, float] = {}
    for agent_id in contributions:
        vals = [float(r.get(agent_id, 0.0)) for r in recent]
        rolling_avg[agent_id] = statistics.fmean(vals) if vals else 0.0

    return {
        "mean_contribution": mean_contrib,
        "median_contribution": median_contrib,
        "total_pot": pot,
        "redistributed_per_agent": redistributed_per_agent,
        "wealth_min": min(wealth_values) if wealth_values else 0.0,
        "wealth_max": max(wealth_values) if wealth_values else 0.0,
        "wealth_mean": mean_wealth,
        "wealth_std": std_wealth,
        "gini": gini(wealth_values),
        "rank_change": rank_change,
        "rolling_contribution_avg": rolling_avg,
    }


def aggregate_runs(run_dirs: List[Path]) -> Dict[str, Any]:
    round_mean_contrib: Dict[int, List[float]] = defaultdict(list)
    wealth_by_agent_round: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    wealth_by_model_round: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    contribution_distribution: List[Dict[str, Any]] = []

    run_index_records: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json", default={}) or {}
        model_assignment = dict(config.get("model_assignment", {}))
        rounds = read_jsonl(run_dir / "rounds.jsonl")

        run_index_records.append(
            {
                "run_id": run_dir.name,
                "rounds": len(rounds),
                "status": (read_json(run_dir / "run_state.json", default={}) or {}).get("status", "unknown"),
            }
        )

        for round_record in rounds:
            round_index = int(round_record.get("round", 0))
            stats = round_record.get("stats", {})
            contributions = {str(k): float(v) for k, v in dict(round_record.get("contributions", {})).items()}
            wealth_after = {str(k): float(v) for k, v in dict(round_record.get("wealth_after", {})).items()}

            mean_contrib = float(stats.get("mean_contribution", 0.0))
            round_mean_contrib[round_index].append(mean_contrib)

            for agent_id, wealth in wealth_after.items():
                wealth_by_agent_round[(agent_id, round_index)].append(wealth)
                model = str(model_assignment.get(agent_id, "unknown"))
                wealth_by_model_round[(model, round_index)].append(wealth)

            for agent_id, contribution in contributions.items():
                contribution_distribution.append(
                    {
                        "run_id": run_dir.name,
                        "round": round_index,
                        "agent_id": agent_id,
                        "model": str(model_assignment.get(agent_id, "unknown")),
                        "contribution": contribution,
                    }
                )

    mean_contribution_rows = [
        {
            "round": round_index,
            "avg_mean_contribution": statistics.fmean(vals) if vals else 0.0,
            "num_runs": len(vals),
        }
        for round_index, vals in sorted(round_mean_contrib.items())
    ]

    wealth_agent_rows = [
        {
            "agent_id": agent_id,
            "round": round_index,
            "avg_wealth": statistics.fmean(vals) if vals else 0.0,
            "num_points": len(vals),
        }
        for (agent_id, round_index), vals in sorted(wealth_by_agent_round.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

    wealth_model_rows = [
        {
            "model": model,
            "round": round_index,
            "avg_wealth": statistics.fmean(vals) if vals else 0.0,
            "num_points": len(vals),
        }
        for (model, round_index), vals in sorted(wealth_by_model_round.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

    return {
        "num_runs": len(run_dirs),
        "runs": run_index_records,
        "average_mean_contribution_per_round": mean_contribution_rows,
        "average_wealth_per_agent_per_round": wealth_agent_rows,
        "average_wealth_per_model_per_round": wealth_model_rows,
        "contribution_distribution": contribution_distribution,
    }

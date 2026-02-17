
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GameSettings:
    agents: int = 6
    rounds: int = 20
    initial_wealth: float = 100.0
    multiplier: float = 1.5
    contribution_history_window: int = 3
    rolling_window: int = 3
    chat_history_messages: int = 120


@dataclass
class LlmSettings:
    default_model: str = "llama3.1:8b"
    models: List[str] = field(default_factory=list)
    random_assign: bool = False
    agent_model_map: Dict[str, str] = field(default_factory=dict)
    summarizer_model: Optional[str] = None
    max_job_retries: int = 2
    parse_repair_retries: int = 1
    system_prompt_message: str = (
        "You are one player in a repeated public-goods game. Be strategic and concise."
    )
    system_prompt_contribution: str = (
        "You are one player in a repeated public-goods game. Output strict JSON only."
    )
    system_prompt_summarizer: str = (
        "Summarize this game round in concise bullets and one short stats line."
    )


@dataclass
class SlurmSettings:
    partition: str = "cuda"
    gres: str = "gpu:1"
    cpus_per_gpu: int = 12
    mem: str = "64G"
    time_limit: str = "00:15:00"
    conda_env: str = "llm_contrib_game"
    poll_interval: int = 5


@dataclass
class RuntimeSettings:
    data_root: Path = Path("data/runs")
    num_runs: int = 1
    seed: int = 123


@dataclass
class ResolvedConfig:
    game: GameSettings
    llm: LlmSettings
    slurm: SlurmSettings
    runtime: RuntimeSettings


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    raise ValueError(f"Expected string or list for model list, got {type(value)}")


def agent_ids(agent_count: int) -> List[str]:
    return [f"agent_{idx + 1}" for idx in range(agent_count)]


def parse_resolved_config(payload: Dict[str, Any]) -> ResolvedConfig:
    game_data = payload.get("game", {})
    llm_data = payload.get("llm", {})
    slurm_data = payload.get("slurm", {})
    runtime_data = payload.get("runtime", {})

    game = GameSettings(
        agents=_as_int(game_data.get("agents"), 6),
        rounds=_as_int(game_data.get("rounds"), 20),
        initial_wealth=_as_float(game_data.get("initial_wealth"), 100.0),
        multiplier=_as_float(game_data.get("multiplier"), 1.5),
        contribution_history_window=_as_int(game_data.get("contribution_history_window"), 3),
        rolling_window=_as_int(game_data.get("rolling_window"), 3),
        chat_history_messages=_as_int(game_data.get("chat_history_messages"), 120),
    )

    agent_model_map_raw = llm_data.get("agent_model_map")
    if agent_model_map_raw is None:
        agent_model_map_raw = {}
    if not isinstance(agent_model_map_raw, dict):
        raise ValueError("llm.agent_model_map must be a mapping if provided")

    llm = LlmSettings(
        default_model=str(llm_data.get("default_model", "llama3.1:8b")),
        models=_as_str_list(llm_data.get("models", [])),
        random_assign=_as_bool(llm_data.get("random_assign"), False),
        agent_model_map={str(k): str(v) for k, v in agent_model_map_raw.items()},
        summarizer_model=(
            None if llm_data.get("summarizer_model") in {None, ""} else str(llm_data.get("summarizer_model"))
        ),
        max_job_retries=_as_int(llm_data.get("max_job_retries"), 2),
        parse_repair_retries=_as_int(llm_data.get("parse_repair_retries"), 1),
        system_prompt_message=str(
            llm_data.get(
                "system_prompt_message",
                "You are one player in a repeated public-goods game. Be strategic and concise.",
            )
        ),
        system_prompt_contribution=str(
            llm_data.get(
                "system_prompt_contribution",
                "You are one player in a repeated public-goods game. Output strict JSON only.",
            )
        ),
        system_prompt_summarizer=str(
            llm_data.get(
                "system_prompt_summarizer",
                "Summarize this game round in concise bullets and one short stats line.",
            )
        ),
    )

    slurm = SlurmSettings(
        partition=str(slurm_data.get("partition", "cuda")),
        gres=str(slurm_data.get("gres", "gpu:1")),
        cpus_per_gpu=_as_int(slurm_data.get("cpus_per_gpu"), 12),
        mem=str(slurm_data.get("mem", "64G")),
        time_limit=str(slurm_data.get("time_limit", "00:15:00")),
        conda_env=str(slurm_data.get("conda_env", "llm_contrib_game")),
        poll_interval=_as_int(slurm_data.get("poll_interval"), 5),
    )

    runtime = RuntimeSettings(
        data_root=Path(str(runtime_data.get("data_root", "data/runs"))),
        num_runs=_as_int(runtime_data.get("num_runs"), 1),
        seed=_as_int(runtime_data.get("seed"), 123),
    )

    if game.agents <= 1:
        raise ValueError("game.agents must be > 1")
    if game.rounds <= 0:
        raise ValueError("game.rounds must be > 0")
    if game.initial_wealth < 0:
        raise ValueError("game.initial_wealth must be >= 0")
    if not (0.0 < game.multiplier):
        raise ValueError("game.multiplier must be > 0")
    if llm.max_job_retries < 0:
        raise ValueError("llm.max_job_retries must be >= 0")
    if llm.parse_repair_retries < 0:
        raise ValueError("llm.parse_repair_retries must be >= 0")

    return ResolvedConfig(game=game, llm=llm, slurm=slurm, runtime=runtime)


def resolve_model_assignments(
    *,
    agent_ids_in_order: List[str],
    llm_settings: LlmSettings,
    run_seed: int,
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}

    for agent_id in agent_ids_in_order:
        if agent_id in llm_settings.agent_model_map:
            resolved[agent_id] = llm_settings.agent_model_map[agent_id]

    unassigned = [agent_id for agent_id in agent_ids_in_order if agent_id not in resolved]
    pool = llm_settings.models if llm_settings.models else [llm_settings.default_model]

    if llm_settings.random_assign:
        rng = random.Random(run_seed)
        for agent_id in unassigned:
            resolved[agent_id] = rng.choice(pool)
    else:
        for idx, agent_id in enumerate(unassigned):
            resolved[agent_id] = pool[idx % len(pool)]

    return resolved


def config_to_dict(config: ResolvedConfig) -> Dict[str, Any]:
    return {
        "game": {
            "agents": config.game.agents,
            "rounds": config.game.rounds,
            "initial_wealth": config.game.initial_wealth,
            "multiplier": config.game.multiplier,
            "contribution_history_window": config.game.contribution_history_window,
            "rolling_window": config.game.rolling_window,
            "chat_history_messages": config.game.chat_history_messages,
        },
        "llm": {
            "default_model": config.llm.default_model,
            "models": config.llm.models,
            "random_assign": config.llm.random_assign,
            "agent_model_map": config.llm.agent_model_map,
            "summarizer_model": config.llm.summarizer_model,
            "max_job_retries": config.llm.max_job_retries,
            "parse_repair_retries": config.llm.parse_repair_retries,
            "system_prompt_message": config.llm.system_prompt_message,
            "system_prompt_contribution": config.llm.system_prompt_contribution,
            "system_prompt_summarizer": config.llm.system_prompt_summarizer,
        },
        "slurm": {
            "partition": config.slurm.partition,
            "gres": config.slurm.gres,
            "cpus_per_gpu": config.slurm.cpus_per_gpu,
            "mem": config.slurm.mem,
            "time_limit": config.slurm.time_limit,
            "conda_env": config.slurm.conda_env,
            "poll_interval": config.slurm.poll_interval,
        },
        "runtime": {
            "data_root": str(config.runtime.data_root),
            "num_runs": config.runtime.num_runs,
            "seed": config.runtime.seed,
        },
    }


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.game import ResolvedConfig, agent_ids, config_to_dict, resolve_model_assignments
from src.io import (
    append_jsonl,
    ensure_dir,
    generate_run_id,
    read_json,
    utc_now_iso,
    write_csv,
    write_json,
)
from src.llm_client_hpc import HpcLlmClient, SlurmOptions, build_slurm_options, ensure_slurm_commands
from src.prompts import (
    build_contribution_prompt,
    build_contribution_repair_prompt,
    build_message_prompt,
    build_round_summarizer_prompt,
    parse_contribution_response,
    parse_message_response,
    recent_contributions_for_agent,
)
from src.stats import aggregate_runs, compute_round_stats


@dataclass
class RunExecutionResult:
    run_id: str
    run_dir: Path
    status: str
    model_assignment: Dict[str, str]
    error: Optional[str] = None


ParseFn = Callable[[str], Tuple[Optional[Dict[str, Any]], Optional[str]]]


def _format_agent_component(agent_id: Optional[str]) -> str:
    if not agent_id:
        return "global"
    return agent_id.replace("/", "_")


def _chat_slice(chat_history: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return list(chat_history)
    return list(chat_history[-limit:])


def _log_event(
    *,
    events_path: Path,
    run_id: str,
    round_index: int,
    subround: str,
    agent_id: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    parsed_result: Optional[Dict[str, Any]],
    parse_error: Optional[str],
    attempt: int,
    call_kind: str,
    job_meta: Dict[str, Any],
) -> None:
    append_jsonl(
        events_path,
        {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "round": round_index,
            "subround": subround,
            "call_kind": call_kind,
            "agent_id": agent_id,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_result": parsed_result,
            "parse_error": parse_error,
            "attempt": attempt,
            "job": job_meta,
        },
    )


def _run_call_with_retries(
    *,
    client: HpcLlmClient,
    run_dir: Path,
    run_id: str,
    events_path: Path,
    round_index: int,
    subround: str,
    call_kind: str,
    agent_id: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_job_retries: int,
    parser: Optional[ParseFn] = None,
) -> Dict[str, Any]:
    errors: List[str] = []

    for attempt in range(1, max_job_retries + 2):
        call_dir = (
            run_dir
            / "jobs"
            / f"round_{round_index:03d}"
            / subround
            / _format_agent_component(agent_id)
            / call_kind
            / f"attempt_{attempt:02d}"
        )

        turn_result = client.infer(
            run_dir=call_dir,
            model=model,
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        parsed_result: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        raw_response = turn_result.answer_text if turn_result.success else ""

        if turn_result.success:
            if parser is not None:
                parsed_result, parse_error = parser(raw_response)
            else:
                parsed_result = {"text": raw_response.strip()}
        else:
            parse_error = turn_result.error_message or f"job_failed_state:{turn_result.state}"

        job_meta = {
            "job_id": turn_result.job_id,
            "status": turn_result.state,
            "success": turn_result.success,
            "run_dir": str(turn_result.run_dir),
            "stdout_log": str(turn_result.stdout_log),
            "stderr_log": str(turn_result.stderr_log),
            "answer_file": str(turn_result.answer_file),
            "error_message": turn_result.error_message,
        }

        _log_event(
            events_path=events_path,
            run_id=run_id,
            round_index=round_index,
            subround=subround,
            agent_id=agent_id,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_response,
            parsed_result=parsed_result,
            parse_error=parse_error,
            attempt=attempt,
            call_kind=call_kind,
            job_meta=job_meta,
        )

        if turn_result.success:
            return {
                "raw_response": raw_response,
                "parsed_result": parsed_result,
                "parse_error": parse_error,
                "job": job_meta,
            }

        errors.append(parse_error or f"attempt_{attempt}_failed")

    raise RuntimeError(
        f"LLM job failed after {max_job_retries + 1} attempts "
        f"(round={round_index}, subround={subround}, agent={agent_id}). "
        f"Errors: {' | '.join(errors)}"
    )


def _make_message_parser() -> ParseFn:
    def _parser(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        message, parse_error = parse_message_response(raw)
        if message is None:
            return None, parse_error or "message_parse_failed"
        return {"message": message}, parse_error

    return _parser


def _make_contribution_parser(wealth: float) -> ParseFn:
    def _parser(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        contribution, reason, parse_error = parse_contribution_response(raw, wealth)
        if parse_error is not None or contribution is None:
            return None, parse_error or "contribution_parse_failed"
        return {"contribution": contribution, "reason": reason or ""}, None

    return _parser


def _resolve_contribution(
    *,
    client: HpcLlmClient,
    run_dir: Path,
    run_id: str,
    events_path: Path,
    round_index: int,
    agent_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_job_retries: int,
    parse_repair_retries: int,
    wealth: float,
) -> Tuple[float, str, Optional[str]]:
    parser = _make_contribution_parser(wealth)

    primary = _run_call_with_retries(
        client=client,
        run_dir=run_dir,
        run_id=run_id,
        events_path=events_path,
        round_index=round_index,
        subround="contribution",
        call_kind="primary",
        agent_id=agent_id,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_job_retries=max_job_retries,
        parser=parser,
    )

    parsed = primary.get("parsed_result") or {}
    if "contribution" in parsed:
        return float(parsed["contribution"]), str(parsed.get("reason", "")), primary.get("parse_error")

    parse_errors = [primary.get("parse_error") or "contribution_parse_failed"]
    raw_to_repair = primary.get("raw_response", "")

    for repair_idx in range(1, parse_repair_retries + 1):
        repair_prompt = build_contribution_repair_prompt(wealth=wealth, raw_response=raw_to_repair)
        try:
            repair = _run_call_with_retries(
                client=client,
                run_dir=run_dir,
                run_id=run_id,
                events_path=events_path,
                round_index=round_index,
                subround="contribution",
                call_kind=f"repair_{repair_idx:02d}",
                agent_id=agent_id,
                model=model,
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_job_retries=max_job_retries,
                parser=parser,
            )
        except RuntimeError as exc:
            parse_errors.append(f"repair_job_failed:{exc}")
            continue

        parsed_repair = repair.get("parsed_result") or {}
        if "contribution" in parsed_repair:
            merged_error = ";".join([x for x in parse_errors if x]) or repair.get("parse_error")
            return (
                float(parsed_repair["contribution"]),
                str(parsed_repair.get("reason", "")),
                merged_error,
            )

        parse_errors.append(repair.get("parse_error") or f"repair_{repair_idx}_parse_failed")
        raw_to_repair = repair.get("raw_response", raw_to_repair)

    fallback_parse_error = ";".join([x for x in parse_errors if x])

    _log_event(
        events_path=events_path,
        run_id=run_id,
        round_index=round_index,
        subround="contribution",
        call_kind="fallback_zero",
        agent_id=agent_id,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw_to_repair,
        parsed_result={"contribution": 0.0, "reason": "fallback_zero_after_parse_failure"},
        parse_error=fallback_parse_error,
        attempt=0,
        job_meta={
            "job_id": "N/A",
            "status": "PARSE_FALLBACK",
            "success": True,
            "run_dir": "N/A",
            "stdout_log": "N/A",
            "stderr_log": "N/A",
            "answer_file": "N/A",
            "error_message": None,
        },
    )

    return 0.0, "fallback_zero_after_parse_failure", fallback_parse_error


def _write_run_state(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at"] = utc_now_iso()
    write_json(path, payload)


def run_single_game(
    *,
    config: ResolvedConfig,
    run_index: int,
    run_seed: int,
    run_id: Optional[str] = None,
) -> RunExecutionResult:
    run_id_resolved = run_id or generate_run_id(prefix=f"run{run_index + 1:03d}")
    run_dir = ensure_dir(config.runtime.data_root / run_id_resolved)
    ensure_dir(run_dir / "jobs")

    paths = {
        "config": run_dir / "config.json",
        "events": run_dir / "events.jsonl",
        "rounds": run_dir / "rounds.jsonl",
        "round_results": run_dir / "round_results.jsonl",
        "summaries": run_dir / "summaries.jsonl",
        "state": run_dir / "run_state.json",
    }

    agents = agent_ids(config.game.agents)
    model_assignment = resolve_model_assignments(
        agent_ids_in_order=agents,
        llm_settings=config.llm,
        run_seed=run_seed,
    )

    summarizer_model = config.llm.summarizer_model or config.llm.default_model

    run_config = config_to_dict(config)
    run_config["run_id"] = run_id_resolved
    run_config["run_index"] = run_index
    run_config["run_seed"] = run_seed
    run_config["model_assignment"] = model_assignment
    run_config["summarizer_model"] = summarizer_model
    write_json(paths["config"], run_config)

    state_payload = {
        "run_id": run_id_resolved,
        "status": "running",
        "started_at": utc_now_iso(),
        "current_round": 0,
        "total_rounds": config.game.rounds,
        "agents": agents,
        "model_assignment": model_assignment,
    }
    _write_run_state(paths["state"], state_payload)

    slurm_options: SlurmOptions = build_slurm_options(
        partition=config.slurm.partition,
        gres=config.slurm.gres,
        cpus_per_gpu=config.slurm.cpus_per_gpu,
        mem=config.slurm.mem,
        time_limit=config.slurm.time_limit,
        conda_env=config.slurm.conda_env,
        poll_interval=config.slurm.poll_interval,
    )
    client = HpcLlmClient(slurm_options=slurm_options, submit_cwd=Path.cwd())

    wealth = {agent_id: float(config.game.initial_wealth) for agent_id in agents}
    contribution_history: List[Dict[str, float]] = []
    chat_history: List[Dict[str, Any]] = []

    try:
        ensure_slurm_commands()

        for round_index in range(1, config.game.rounds + 1):
            wealth_before = dict(wealth)
            round_messages: Dict[str, str] = {}

            if round_index >= 2:
                for agent_id in agents:
                    recent_other = recent_contributions_for_agent(
                        agent_id=agent_id,
                        contribution_history=contribution_history,
                        window=config.game.contribution_history_window,
                    )
                    message_prompt = build_message_prompt(
                        agent_id=agent_id,
                        wealth=wealth[agent_id],
                        round_index=round_index,
                        recent_other_contributions=recent_other,
                        chat_history=_chat_slice(chat_history, config.game.chat_history_messages),
                    )

                    message_call = _run_call_with_retries(
                        client=client,
                        run_dir=run_dir,
                        run_id=run_id_resolved,
                        events_path=paths["events"],
                        round_index=round_index,
                        subround="message",
                        call_kind="primary",
                        agent_id=agent_id,
                        model=model_assignment[agent_id],
                        system_prompt=config.llm.system_prompt_message,
                        user_prompt=message_prompt,
                        max_job_retries=config.llm.max_job_retries,
                        parser=_make_message_parser(),
                    )

                    parsed_message = (message_call.get("parsed_result") or {}).get("message", "")
                    round_messages[agent_id] = str(parsed_message)

                    chat_history.append(
                        {
                            "timestamp": utc_now_iso(),
                            "round": round_index,
                            "agent_id": agent_id,
                            "message": round_messages[agent_id],
                        }
                    )

            contributions: Dict[str, float] = {}
            contribution_reasons: Dict[str, str] = {}
            contribution_parse_errors: Dict[str, Optional[str]] = {}

            for agent_id in agents:
                recent_other = recent_contributions_for_agent(
                    agent_id=agent_id,
                    contribution_history=contribution_history,
                    window=config.game.contribution_history_window,
                )
                contribution_prompt = build_contribution_prompt(
                    agent_id=agent_id,
                    wealth=wealth[agent_id],
                    round_index=round_index,
                    round_messages=round_messages,
                    recent_other_contributions=recent_other,
                )

                contribution_value, contribution_reason, parse_error = _resolve_contribution(
                    client=client,
                    run_dir=run_dir,
                    run_id=run_id_resolved,
                    events_path=paths["events"],
                    round_index=round_index,
                    agent_id=agent_id,
                    model=model_assignment[agent_id],
                    system_prompt=config.llm.system_prompt_contribution,
                    user_prompt=contribution_prompt,
                    max_job_retries=config.llm.max_job_retries,
                    parse_repair_retries=config.llm.parse_repair_retries,
                    wealth=wealth[agent_id],
                )

                contributions[agent_id] = contribution_value
                contribution_reasons[agent_id] = contribution_reason
                contribution_parse_errors[agent_id] = parse_error

            contribution_history.append(dict(contributions))

            pot = float(sum(contributions.values()))
            redistributed_per_agent = float(config.game.multiplier * pot / len(agents))

            for agent_id in agents:
                wealth[agent_id] = (wealth[agent_id] - contributions[agent_id]) + redistributed_per_agent

            round_stats = compute_round_stats(
                contributions=contributions,
                wealth_before=wealth_before,
                wealth_after=wealth,
                pot=pot,
                redistributed_per_agent=redistributed_per_agent,
                contribution_history=contribution_history,
                rolling_window=config.game.rolling_window,
            )

            summary_prompt = build_round_summarizer_prompt(
                round_index=round_index,
                messages=round_messages,
                contributions=contributions,
                wealth_before=wealth_before,
                wealth_after=wealth,
                pot=pot,
                redistributed_per_agent=redistributed_per_agent,
                gini=float(round_stats["gini"]),
                mean_contribution=float(round_stats["mean_contribution"]),
            )

            summary_call = _run_call_with_retries(
                client=client,
                run_dir=run_dir,
                run_id=run_id_resolved,
                events_path=paths["events"],
                round_index=round_index,
                subround="summary",
                call_kind="primary",
                agent_id="summarizer",
                model=summarizer_model,
                system_prompt=config.llm.system_prompt_summarizer,
                user_prompt=summary_prompt,
                max_job_retries=config.llm.max_job_retries,
                parser=None,
            )

            summary_text = summary_call.get("raw_response", "").strip()
            summary_record = {
                "timestamp": utc_now_iso(),
                "run_id": run_id_resolved,
                "round": round_index,
                "model": summarizer_model,
                "summary_text": summary_text,
            }
            append_jsonl(paths["summaries"], summary_record)

            round_record = {
                "timestamp": utc_now_iso(),
                "run_id": run_id_resolved,
                "round": round_index,
                "messages": round_messages,
                "contributions": contributions,
                "contribution_reasons": contribution_reasons,
                "contribution_parse_errors": contribution_parse_errors,
                "wealth_before": wealth_before,
                "wealth_after": dict(wealth),
                "pot": pot,
                "redistributed_per_agent": redistributed_per_agent,
                "stats": round_stats,
                "summary": summary_text,
            }
            append_jsonl(paths["rounds"], round_record)
            append_jsonl(paths["round_results"], round_record)

            state_payload["current_round"] = round_index
            state_payload["latest_round"] = round_record
            state_payload["latest_summary"] = summary_record
            _write_run_state(paths["state"], state_payload)

        state_payload["status"] = "completed"
        state_payload["completed_at"] = utc_now_iso()
        state_payload["final_wealth"] = wealth
        _write_run_state(paths["state"], state_payload)

        return RunExecutionResult(
            run_id=run_id_resolved,
            run_dir=run_dir,
            status="completed",
            model_assignment=model_assignment,
            error=None,
        )
    except KeyboardInterrupt:
        state_payload["status"] = "interrupted"
        state_payload["interrupted_at"] = utc_now_iso()
        state_payload["error"] = "KeyboardInterrupt"
        _write_run_state(paths["state"], state_payload)

        return RunExecutionResult(
            run_id=run_id_resolved,
            run_dir=run_dir,
            status="interrupted",
            model_assignment=model_assignment,
            error="KeyboardInterrupt",
        )
    except Exception as exc:
        state_payload["status"] = "failed"
        state_payload["failed_at"] = utc_now_iso()
        state_payload["error"] = str(exc)
        _write_run_state(paths["state"], state_payload)

        return RunExecutionResult(
            run_id=run_id_resolved,
            run_dir=run_dir,
            status="failed",
            model_assignment=model_assignment,
            error=str(exc),
        )


def run_experiment(
    *,
    config: ResolvedConfig,
    num_runs: int,
    base_seed: int,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    if num_runs <= 0:
        raise ValueError("num_runs must be > 0")

    experiment_root = ensure_dir(Path("data/experiments"))
    experiment_id_resolved = experiment_id or generate_run_id(prefix="exp")
    experiment_dir = ensure_dir(experiment_root / experiment_id_resolved)

    run_results: List[RunExecutionResult] = []
    interrupted = False

    for run_index in range(num_runs):
        run_seed = base_seed + run_index
        try:
            result = run_single_game(
                config=config,
                run_index=run_index,
                run_seed=run_seed,
                run_id=None,
            )
            run_results.append(result)
            if result.status == "interrupted":
                interrupted = True
                break
        except Exception as exc:
            failed_run_id = generate_run_id(prefix=f"run{run_index + 1:03d}_failed")
            result = RunExecutionResult(
                run_id=failed_run_id,
                run_dir=config.runtime.data_root / failed_run_id,
                status="failed",
                model_assignment={},
                error=str(exc),
            )
            run_results.append(result)

    success_dirs = [r.run_dir for r in run_results if r.status == "completed"]
    aggregate = aggregate_runs(success_dirs)
    aggregate["experiment_id"] = experiment_id_resolved
    aggregate["base_seed"] = base_seed
    aggregate["requested_runs"] = num_runs
    aggregate["completed_runs"] = len(success_dirs)
    aggregate["failed_runs"] = len(run_results) - len(success_dirs)
    aggregate["interrupted"] = interrupted

    run_manifest = [
        {
            "run_id": r.run_id,
            "run_dir": str(r.run_dir),
            "status": r.status,
            "error": r.error,
            "model_assignment": r.model_assignment,
        }
        for r in run_results
    ]

    write_json(experiment_dir / "experiment_config.json", config_to_dict(config))
    write_json(experiment_dir / "runs_manifest.json", {"runs": run_manifest})
    write_json(experiment_dir / "aggregate_summary.json", aggregate)

    write_csv(
        experiment_dir / "average_mean_contribution_per_round.csv",
        aggregate["average_mean_contribution_per_round"],
        ["round", "avg_mean_contribution", "num_runs"],
    )
    write_csv(
        experiment_dir / "average_wealth_per_agent_per_round.csv",
        aggregate["average_wealth_per_agent_per_round"],
        ["agent_id", "round", "avg_wealth", "num_points"],
    )
    write_csv(
        experiment_dir / "average_wealth_per_model_per_round.csv",
        aggregate["average_wealth_per_model_per_round"],
        ["model", "round", "avg_wealth", "num_points"],
    )
    write_csv(
        experiment_dir / "contribution_distribution.csv",
        aggregate["contribution_distribution"],
        ["run_id", "round", "agent_id", "model", "contribution"],
    )

    latest_run_dir = read_json(experiment_dir / "latest_run_dir.json", default={})
    latest_run_dir = latest_run_dir if isinstance(latest_run_dir, dict) else {}
    write_json(
        experiment_dir / "latest_run_dir.json",
        {
            **latest_run_dir,
            "latest_run_dir": str(run_results[-1].run_dir) if run_results else None,
            "updated_at": utc_now_iso(),
        },
    )

    return {
        "experiment_id": experiment_id_resolved,
        "experiment_dir": str(experiment_dir),
        "interrupted": interrupted,
        "runs": run_manifest,
        "aggregate_summary_path": str(experiment_dir / "aggregate_summary.json"),
    }

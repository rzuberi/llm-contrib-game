
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_PARTITION = "cuda"
DEFAULT_GRES = "gpu:1"
DEFAULT_CPUS_PER_GPU = 12
DEFAULT_MEM = "64G"
DEFAULT_TIME_LIMIT = "00:15:00"
DEFAULT_ENV_NAME = "llm_contrib_game"
DEFAULT_POLL_INTERVAL = 5

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}

FAILURE_STATES = TERMINAL_STATES - {"COMPLETED"}

# Copied from ../hpc-llm/llm/slurm.py and adapted for this project.
SBATCH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

CONDA_BASE_HINT={{ conda_base_hint }}
CONDA_EXE_HINT={{ conda_exe_hint }}

if [ -n "$CONDA_BASE_HINT" ] && [ -f "$CONDA_BASE_HINT/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE_HINT/etc/profile.d/conda.sh"
elif [ -n "$CONDA_EXE_HINT" ] && [ -x "$CONDA_EXE_HINT" ]; then
  CONDA_BASE_FROM_EXE="$($CONDA_EXE_HINT info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE_FROM_EXE" ] && [ -f "$CONDA_BASE_FROM_EXE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE_FROM_EXE/etc/profile.d/conda.sh"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "$candidate/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  log "ERROR: conda not found on compute node. Set LLM_CONDA_BASE or ensure conda is available in non-interactive SLURM jobs."
  exit 1
fi

conda activate {{ conda_env }}

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA_BIN="$(command -v ollama || true)"
if [ -z "$OLLAMA_BIN" ] && [ -x "$HOME/.local/bin/ollama" ]; then
  OLLAMA_BIN="$HOME/.local/bin/ollama"
fi
if [ -z "$OLLAMA_BIN" ]; then
  log "ERROR: ollama not found in PATH. Install user-space ollama first."
  exit 1
fi

export OLLAMA_HOST=127.0.0.1:11434
if [ -z "${OLLAMA_MODELS:-}" ]; then
  user_name="$(whoami)"
  group_name="$(id -gn 2>/dev/null || true)"

  pick_models_dir() {
    local base candidate
    if [ -n "${SCRATCH:-}" ]; then
      candidate="$SCRATCH/.ollama/models"
      if mkdir -p "$candidate" >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi

    for base in /mnt/scratchc /scratchc /mnt/scratch /scratch; do
      [ -d "$base" ] || continue
      for candidate in \
        "$base/$user_name/.ollama/models" \
        "$base/$group_name/$user_name/.ollama/models"; do
        if mkdir -p "$candidate" >/dev/null 2>&1; then
          echo "$candidate"
          return 0
        fi
      done
    done

    candidate="$HOME/.ollama/models"
    mkdir -p "$candidate"
    echo "$candidate"
  }

  OLLAMA_MODELS="$(pick_models_dir)"
  export OLLAMA_MODELS
fi
mkdir -p "$OLLAMA_MODELS"

REQUEST_FILE={{ request_file }}
PROMPT_FILE={{ prompt_file }}
SYSTEM_FILE={{ system_file }}
ANSWER_FILE={{ answer_file }}

"$OLLAMA_BIN" serve >/dev/null 2>&1 &
OLLAMA_PID="$!"

cleanup() {
  local rc=$?
  if kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 90); do
  if "$OLLAMA_BIN" list >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  log "ERROR: ollama server did not become ready within 90 seconds."
  exit 1
fi

MODEL="$(python - "$REQUEST_FILE" "$PROMPT_FILE" "$SYSTEM_FILE" <<'PY'
import json
import pathlib
import sys

request_path = pathlib.Path(sys.argv[1])
prompt_path = pathlib.Path(sys.argv[2])
system_path = pathlib.Path(sys.argv[3])

data = json.loads(request_path.read_text(encoding="utf-8"))
prompt_path.write_text(data.get("prompt", ""), encoding="utf-8")
system_path.write_text(data.get("system_prompt", ""), encoding="utf-8")
print(data.get("model", ""))
PY
)"
if [ -z "$MODEL" ]; then
  log "ERROR: request.json does not contain a model value."
  exit 1
fi

if ! "$OLLAMA_BIN" show "$MODEL" >/dev/null 2>&1; then
  log "Model '$MODEL' not present locally. Pulling..."
  if ! "$OLLAMA_BIN" pull "$MODEL"; then
    log "ERROR: failed to pull model '$MODEL'. Check network access and model name."
    exit 1
  fi
fi

PROMPT_CONTENT="$(cat "$PROMPT_FILE")"
SYSTEM_CONTENT="$(cat "$SYSTEM_FILE")"
RUN_HELP="$("$OLLAMA_BIN" run --help 2>&1 || true)"

if [ -n "$SYSTEM_CONTENT" ]; then
  if printf '%s\n' "$RUN_HELP" | grep -q -- '--system'; then
    "$OLLAMA_BIN" run "$MODEL" --system "$SYSTEM_CONTENT" "$PROMPT_CONTENT" >"$ANSWER_FILE"
  else
    log "ollama run does not support --system on this node; injecting system prompt into user prompt."
    COMBINED_PROMPT="$(printf 'System instructions:\n%s\n\nUser prompt:\n%s\n' "$SYSTEM_CONTENT" "$PROMPT_CONTENT")"
    "$OLLAMA_BIN" run "$MODEL" "$COMBINED_PROMPT" >"$ANSWER_FILE"
  fi
else
  "$OLLAMA_BIN" run "$MODEL" "$PROMPT_CONTENT" >"$ANSWER_FILE"
fi

log "Inference completed."
"""


class RequestPayload(BaseModel):
    model: str = Field(min_length=1)
    prompt: str
    system_prompt: str
    created_at_utc: str


class MetaPayload(BaseModel):
    user: str
    submit_host: str
    submit_cwd: str
    slurm_job_id: str
    started_at_utc: str
    ended_at_utc: str
    slurm_state: str
    slurm_partition: str
    slurm_gres: str
    slurm_cpus_per_gpu: int
    slurm_mem: str
    slurm_time: str
    stdout_log: str
    stderr_log: str
    run_dir: str
    sacct_state: Optional[str] = None
    sacct_start: Optional[str] = None
    sacct_end: Optional[str] = None
    sacct_exit_code: Optional[str] = None


@dataclass
class SlurmOptions:
    partition: str = DEFAULT_PARTITION
    gres: str = DEFAULT_GRES
    cpus_per_gpu: int = DEFAULT_CPUS_PER_GPU
    mem: str = DEFAULT_MEM
    time_limit: str = DEFAULT_TIME_LIMIT
    conda_env: str = DEFAULT_ENV_NAME
    poll_interval: int = DEFAULT_POLL_INTERVAL


@dataclass
class TurnFiles:
    request_file: Path
    meta_file: Path
    sbatch_file: Path
    stdout_log: Path
    stderr_log: Path
    answer_file: Path
    prompt_file: Path
    system_file: Path


@dataclass
class TurnResult:
    run_dir: Path
    job_id: str
    state: str
    success: bool
    stdout_log: Path
    stderr_log: Path
    answer_file: Path
    answer_text: str = ""
    error_message: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Union[BaseModel, Dict[str, Any]]) -> None:
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def ensure_slurm_command(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise RuntimeError(f"Required command '{cmd}' not found in PATH.")


def ensure_slurm_commands() -> None:
    ensure_slurm_command("sbatch")
    ensure_slurm_command("squeue")


def parse_int_env(name: str, fallback: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return fallback
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be > 0.")
    return parsed


def normalize_state(state: str) -> str:
    base = state.strip().split()[0].upper()
    if not base:
        return ""
    if "+" in base:
        base = base.split("+", 1)[0]
    return base


def parse_job_id(sbatch_stdout: str) -> str:
    text = sbatch_stdout.strip()
    if not text:
        raise RuntimeError("sbatch returned empty output; cannot determine job id.")
    first_line = text.splitlines()[0].strip()
    if not first_line:
        raise RuntimeError("sbatch returned malformed output; cannot determine job id.")

    candidate = first_line.split(";", 1)[0].strip()
    if candidate.isdigit():
        return candidate

    match = re.search(r"\d+", candidate)
    if match:
        return match.group(0)
    raise RuntimeError(f"Unable to parse job id from sbatch output: {first_line}")


def query_squeue_state(job_id: str) -> Optional[str]:
    proc = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return normalize_state(lines[0])


def query_sacct_row(job_id: str) -> Optional[Dict[str, str]]:
    if shutil.which("sacct") is None:
        return None

    proc = subprocess.run(
        ["sacct", "-n", "-P", "-X", "-j", job_id, "--format=JobIDRaw,State,Start,End,ExitCode"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None

    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("|")
        if len(parts) < 5:
            continue
        row_job_id, state, start, end, exit_code = parts[:5]
        if row_job_id != job_id:
            continue
        return {
            "state": normalize_state(state),
            "start": start,
            "end": end,
            "exit_code": exit_code,
        }
    return None


def query_job_state(job_id: str) -> Optional[str]:
    squeue_state = query_squeue_state(job_id)
    if squeue_state:
        return squeue_state

    sacct_row = query_sacct_row(job_id)
    if sacct_row and sacct_row.get("state"):
        return sacct_row["state"]
    return None


def wait_for_job(job_id: str, poll_interval: int) -> str:
    missing_count = 0
    while True:
        state = query_job_state(job_id)
        if state:
            missing_count = 0
            if state in TERMINAL_STATES:
                return state
            time.sleep(poll_interval)
            continue

        missing_count += 1
        if missing_count >= 120:
            raise RuntimeError(
                "Unable to determine SLURM job state via squeue/sacct for 120 polls. "
                "Cluster accounting may be delayed."
            )
        time.sleep(poll_interval)


def build_slurm_options(
    partition: Optional[str] = None,
    gres: Optional[str] = None,
    cpus_per_gpu: Optional[int] = None,
    mem: Optional[str] = None,
    time_limit: Optional[str] = None,
    conda_env: Optional[str] = None,
    poll_interval: Optional[int] = None,
) -> SlurmOptions:
    slurm_partition = partition or os.getenv("LLM_PARTITION", DEFAULT_PARTITION)
    slurm_gres = gres or os.getenv("LLM_GRES", DEFAULT_GRES)

    if cpus_per_gpu is None:
        slurm_cpus_per_gpu = parse_int_env("LLM_CPUS_PER_GPU", DEFAULT_CPUS_PER_GPU)
    else:
        if cpus_per_gpu <= 0:
            raise ValueError("cpus_per_gpu must be > 0")
        slurm_cpus_per_gpu = cpus_per_gpu

    slurm_mem = mem or os.getenv("LLM_MEM", DEFAULT_MEM)
    slurm_time_limit = time_limit or os.getenv("LLM_TIME", DEFAULT_TIME_LIMIT)
    slurm_conda_env = conda_env or os.getenv("LLM_CONDA_ENV", DEFAULT_ENV_NAME)

    if poll_interval is None:
        poll_seconds = parse_int_env("LLM_POLL_INTERVAL", DEFAULT_POLL_INTERVAL)
    else:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        poll_seconds = poll_interval

    return SlurmOptions(
        partition=slurm_partition,
        gres=slurm_gres,
        cpus_per_gpu=slurm_cpus_per_gpu,
        mem=slurm_mem,
        time_limit=slurm_time_limit,
        conda_env=slurm_conda_env,
        poll_interval=poll_seconds,
    )


def prepare_turn_files(run_dir: Path) -> TurnFiles:
    run_dir.mkdir(parents=True, exist_ok=True)
    return TurnFiles(
        request_file=run_dir / "request.json",
        meta_file=run_dir / "meta.json",
        sbatch_file=run_dir / "sbatch.sh",
        stdout_log=run_dir / "stdout.log",
        stderr_log=run_dir / "stderr.log",
        answer_file=run_dir / "answer.txt",
        prompt_file=run_dir / "prompt.txt",
        system_file=run_dir / "system_prompt.txt",
    )


def render_sbatch_script(
    script_path: Path,
    conda_env: str,
    conda_base_hint: str,
    conda_exe_hint: str,
    request_file: Path,
    prompt_file: Path,
    system_file: Path,
    answer_file: Path,
) -> None:
    template = Template(SBATCH_TEMPLATE, undefined=StrictUndefined)
    content = template.render(
        conda_env=shlex.quote(conda_env),
        conda_base_hint=shlex.quote(conda_base_hint),
        conda_exe_hint=shlex.quote(conda_exe_hint),
        request_file=shlex.quote(str(request_file)),
        prompt_file=shlex.quote(str(prompt_file)),
        system_file=shlex.quote(str(system_file)),
        answer_file=shlex.quote(str(answer_file)),
    )
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)


def _write_meta(
    *,
    files: TurnFiles,
    slurm: SlurmOptions,
    submit_cwd: Path,
    slurm_job_id: str,
    started_at_utc: str,
    ended_at_utc: str,
    slurm_state: str,
    sacct_row: Optional[Dict[str, str]] = None,
) -> None:
    payload = MetaPayload(
        user=getpass.getuser(),
        submit_host=socket.gethostname(),
        submit_cwd=str(submit_cwd),
        slurm_job_id=slurm_job_id,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        slurm_state=slurm_state,
        slurm_partition=slurm.partition,
        slurm_gres=slurm.gres,
        slurm_cpus_per_gpu=slurm.cpus_per_gpu,
        slurm_mem=slurm.mem,
        slurm_time=slurm.time_limit,
        stdout_log=str(files.stdout_log),
        stderr_log=str(files.stderr_log),
        run_dir=str(files.meta_file.parent),
        sacct_state=(sacct_row or {}).get("state"),
        sacct_start=(sacct_row or {}).get("start"),
        sacct_end=(sacct_row or {}).get("end"),
        sacct_exit_code=(sacct_row or {}).get("exit_code"),
    )
    write_json(files.meta_file, payload)


def run_inference_turn(
    *,
    run_dir: Path,
    model: str,
    prompt: str,
    system_prompt: str,
    slurm: SlurmOptions,
    submit_cwd: Path,
    on_submitted: Optional[Callable[[str], None]] = None,
) -> TurnResult:
    files = prepare_turn_files(run_dir)

    submitted_at = utc_now_iso()
    request_payload = RequestPayload(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        created_at_utc=submitted_at,
    )
    write_json(files.request_file, request_payload)
    files.answer_file.touch(exist_ok=True)

    conda_base_hint = os.getenv("LLM_CONDA_BASE", "") or os.getenv("CONDA_PREFIX", "")
    conda_exe_hint = shutil.which("conda") or ""
    render_sbatch_script(
        script_path=files.sbatch_file,
        conda_env=slurm.conda_env,
        conda_base_hint=conda_base_hint,
        conda_exe_hint=conda_exe_hint,
        request_file=files.request_file,
        prompt_file=files.prompt_file,
        system_file=files.system_file,
        answer_file=files.answer_file,
    )

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "-p",
        slurm.partition,
        "--gres",
        slurm.gres,
        "--cpus-per-gpu",
        str(slurm.cpus_per_gpu),
        "--mem",
        slurm.mem,
        "--time",
        slurm.time_limit,
        "-o",
        str(files.stdout_log),
        "-e",
        str(files.stderr_log),
        str(files.sbatch_file),
    ]

    submit = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=False)
    if submit.returncode != 0:
        ended_at = utc_now_iso()
        error_message = submit.stderr.strip() or "Failed to submit job with sbatch."
        _write_meta(
            files=files,
            slurm=slurm,
            submit_cwd=submit_cwd,
            slurm_job_id="SUBMIT_FAILED",
            started_at_utc=submitted_at,
            ended_at_utc=ended_at,
            slurm_state="SUBMIT_FAILED",
        )
        return TurnResult(
            run_dir=run_dir,
            job_id="SUBMIT_FAILED",
            state="SUBMIT_FAILED",
            success=False,
            stdout_log=files.stdout_log,
            stderr_log=files.stderr_log,
            answer_file=files.answer_file,
            error_message=error_message,
        )

    try:
        job_id = parse_job_id(submit.stdout)
    except RuntimeError as exc:
        ended_at = utc_now_iso()
        _write_meta(
            files=files,
            slurm=slurm,
            submit_cwd=submit_cwd,
            slurm_job_id="SUBMIT_PARSE_ERROR",
            started_at_utc=submitted_at,
            ended_at_utc=ended_at,
            slurm_state="SUBMIT_PARSE_ERROR",
        )
        return TurnResult(
            run_dir=run_dir,
            job_id="SUBMIT_PARSE_ERROR",
            state="SUBMIT_PARSE_ERROR",
            success=False,
            stdout_log=files.stdout_log,
            stderr_log=files.stderr_log,
            answer_file=files.answer_file,
            error_message=str(exc),
        )

    if on_submitted is not None:
        on_submitted(job_id)

    try:
        final_state = wait_for_job(job_id=job_id, poll_interval=slurm.poll_interval)
    except RuntimeError as exc:
        ended_at = utc_now_iso()
        _write_meta(
            files=files,
            slurm=slurm,
            submit_cwd=submit_cwd,
            slurm_job_id=job_id,
            started_at_utc=submitted_at,
            ended_at_utc=ended_at,
            slurm_state="UNKNOWN",
        )
        return TurnResult(
            run_dir=run_dir,
            job_id=job_id,
            state="UNKNOWN",
            success=False,
            stdout_log=files.stdout_log,
            stderr_log=files.stderr_log,
            answer_file=files.answer_file,
            error_message=str(exc),
        )

    sacct_row = query_sacct_row(job_id)
    ended_at = utc_now_iso()
    _write_meta(
        files=files,
        slurm=slurm,
        submit_cwd=submit_cwd,
        slurm_job_id=job_id,
        started_at_utc=submitted_at,
        ended_at_utc=ended_at,
        slurm_state=final_state,
        sacct_row=sacct_row,
    )

    if final_state in FAILURE_STATES:
        return TurnResult(
            run_dir=run_dir,
            job_id=job_id,
            state=final_state,
            success=False,
            stdout_log=files.stdout_log,
            stderr_log=files.stderr_log,
            answer_file=files.answer_file,
            error_message=f"Job ended with state {final_state}",
        )

    if not files.answer_file.exists():
        return TurnResult(
            run_dir=run_dir,
            job_id=job_id,
            state=final_state,
            success=False,
            stdout_log=files.stdout_log,
            stderr_log=files.stderr_log,
            answer_file=files.answer_file,
            error_message=f"answer file missing: {files.answer_file}",
        )

    answer_text = files.answer_file.read_text(encoding="utf-8")
    return TurnResult(
        run_dir=run_dir,
        job_id=job_id,
        state=final_state,
        success=True,
        stdout_log=files.stdout_log,
        stderr_log=files.stderr_log,
        answer_file=files.answer_file,
        answer_text=answer_text,
    )


class HpcLlmClient:
    def __init__(self, slurm_options: Optional[SlurmOptions] = None, submit_cwd: Optional[Path] = None) -> None:
        self.slurm_options = slurm_options or build_slurm_options()
        self.submit_cwd = submit_cwd or Path.cwd()

    def infer(
        self,
        *,
        run_dir: Path,
        model: str,
        prompt: str,
        system_prompt: str,
        on_submitted: Optional[Callable[[str], None]] = None,
    ) -> TurnResult:
        return run_inference_turn(
            run_dir=run_dir,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            slurm=self.slurm_options,
            submit_cwd=self.submit_cwd,
            on_submitted=on_submitted,
        )

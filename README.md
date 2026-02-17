# llm-contrib-game

Self-contained repeated public-goods game where each agent is an LLM call executed as an HPC SLURM job via Ollama.

## What This Implements

- Repeated contribution game with configurable `N` agents and `R` rounds.
- Round 1: contribution only.
- Rounds 2..R:
  - Message sub-round (one message/agent).
  - Contribution sub-round (one contribution/agent).
- Pot update each round:
  - `P = sum(contributions)`
  - `P' = 1.5 * P`
  - each agent receives `P' / N`
  - `wealth_i <- (wealth_i - contribution_i) + P'/N`
- Round summarizer LLM call after wealth update (HPC job).
- Per-call logging for every LLM request/response + SLURM metadata.
- Multi-run experiments with aggregate summaries and CSV outputs.
- Live dashboard that polls JSON state every few seconds.

## HPC Pattern (Copied from `../hpc-llm`)

`src/llm_client_hpc.py` copies/adapts the same submission workflow used in sibling `../hpc-llm/llm/slurm.py`:

- write `request.json` (model, prompt, system prompt)
- render `sbatch.sh` from Jinja template
- `sbatch --parsable ...` submit
- parse job id
- poll via `squeue` and `sacct`
- wait for terminal state
- collect `answer.txt`
- write `meta.json`, `stdout.log`, `stderr.log`

Each LLM call writes files in a per-call run directory with the same style:
`request.json`, `meta.json`, `sbatch.sh`, `stdout.log`, `stderr.log`, `answer.txt`, `prompt.txt`, `system_prompt.txt`.

## Setup

```bash
conda env create -f environment.yml
conda activate llm_contrib_game
```

Requirements outside this repo:

- SLURM commands available (`sbatch`, `squeue`, optionally `sacct`)
- user-space `ollama` binary on compute nodes (same expectation as `../hpc-llm`)

## Run One Game

```bash
python scripts/run_game.py --config configs/default.yaml --num-runs 1
```

Example with explicit options:

```bash
python scripts/run_game.py \
  --rounds 20 \
  --agents 6 \
  --models llama3.1:8b,qwen2.5:14b \
  --random-assign \
  --num-runs 1 \
  --seed 123
```

## Run Many Games

Single entrypoint:

```bash
python scripts/run_game.py --num-runs 10 --seed 123
```

Or wrapper:

```bash
python scripts/run_many.py --num-runs 10 --seed 123 --random-assign --models llama3.1:8b,qwen2.5:14b
```

## Dashboard

Start dashboard:

```bash
python web/app.py --data-root data/runs --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000`

It auto-refreshes and shows:

- current round and run status
- contributions table
- wealth table
- pot / redistributed amount
- latest summarizer output

Compact terminal monitor:

```bash
python scripts/status.py --last 7
python scripts/status.py --watch --interval 5
```

## Output Layout

Per run:

```text
data/runs/<run_id>/
  config.json
  run_state.json
  events.jsonl
  rounds.jsonl
  round_results.jsonl
  summaries.jsonl
  jobs/
    round_001/
    round_002/
    ...
```

`events.jsonl` includes for every LLM call:

- timestamp, run id, round, subround, call type, agent id
- model, system prompt, user prompt
- raw response, parsed result, parse error
- attempt number
- SLURM metadata (`job_id`, status, logs, run_dir, errors)

Experiment aggregates:

```text
data/experiments/<experiment_id>/
  experiment_config.json
  runs_manifest.json
  aggregate_summary.json
  average_mean_contribution_per_round.csv
  average_wealth_per_agent_per_round.csv
  average_wealth_per_model_per_round.csv
  contribution_distribution.csv
```

## Parsing and Robustness

- Contribution outputs are forced to strict JSON schema:
  - `{"contribution": <number>, "reason": "..."}`
- If parsing fails:
  - run repair LLM prompt(s) via HPC jobs
  - if still invalid, fallback to `0.0` contribution (logged)
- Job failures are detected and retried (`llm.max_job_retries` in config).
- The orchestrator does not move to the next step/round unless required jobs complete successfully.

## Configuration

Default config: `configs/default.yaml`

Key controls:

- game size and economics (`agents`, `rounds`, `initial_wealth`, `multiplier`)
- model assignment modes:
  - deterministic map: `llm.agent_model_map`
  - random per-run assignment: `llm.random_assign=true` and `llm.models=[...]`
- SLURM resources (`partition`, `gres`, `cpus_per_gpu`, `mem`, `time_limit`)
- retry controls (`llm.max_job_retries`, `llm.parse_repair_retries`)

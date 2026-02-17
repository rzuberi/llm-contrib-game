#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import find_latest_run_dir, read_json, read_jsonl

app = Flask(__name__, static_folder="static", static_url_path="/static")

DATA_ROOT = Path("data/runs")
PINNED_RUN_ID: Optional[str] = None


def resolve_run_dir() -> Optional[Path]:
    query_run_id = request.args.get("run_id")
    run_id = query_run_id or PINNED_RUN_ID

    if run_id:
        candidate = DATA_ROOT / run_id
        return candidate if candidate.exists() else None

    return find_latest_run_dir(DATA_ROOT)


@app.route("/")
def index() -> object:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def api_state() -> object:
    run_dir = resolve_run_dir()
    if run_dir is None:
        return jsonify({"status": "no_runs_found", "data_root": str(DATA_ROOT.resolve())})

    state = read_json(run_dir / "run_state.json", default={}) or {}
    rounds = read_jsonl(run_dir / "round_results.jsonl")
    summaries = read_jsonl(run_dir / "summaries.jsonl")

    return jsonify(
        {
            "status": "ok",
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "state": state,
            "latest_round": rounds[-1] if rounds else None,
            "latest_summary": summaries[-1] if summaries else None,
            "recent_rounds": rounds[-10:],
        }
    )


def main() -> None:
    global DATA_ROOT, PINNED_RUN_ID

    parser = argparse.ArgumentParser(description="Live dashboard for llm-contrib-game runs.")
    parser.add_argument("--data-root", default="data/runs")
    parser.add_argument("--run-id", default=None, help="Optional run ID to pin in the dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    PINNED_RUN_ID = args.run_id

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

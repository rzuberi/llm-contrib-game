#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_jsonl(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def _event_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except Exception:
        return 0


def _summary_short(run_dir: Path, width: int = 130) -> str:
    latest = _tail_jsonl(run_dir / "summaries.jsonl")
    if latest is None:
        return "-"
    text = " ".join(str(latest.get("summary_text", "")).split())
    if not text:
        return "-"
    return text[:width]


def _stderr_signatures(run_dir: Path, max_files: int = 80) -> List[Tuple[str, int]]:
    patterns = [
        "unknown flag: --system",
        "failed to pull model",
        "ollama server did not become ready",
        "ERROR:",
    ]
    counts = Counter()
    files = sorted(run_dir.rglob("stderr.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:max_files]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        low = content.lower()
        for pat in patterns:
            if pat.lower() in low:
                counts[pat] += 1
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("Error:"):
                counts[line[:120]] += 1
                break
    return counts.most_common(3)


def _squeue_count() -> int:
    try:
        user = os.environ.get("USER", "")
        proc = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%i"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return 0
        return len([ln for ln in proc.stdout.splitlines() if ln.strip()])
    except Exception:
        return 0


def _collect_rows(data_root: Path, last: int) -> List[Dict]:
    runs = [p for p in data_root.glob("run*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for run_dir in runs[:last]:
        state = _read_json(run_dir / "run_state.json")
        rows.append(
            {
                "run_id": run_dir.name,
                "status": state.get("status", "unknown"),
                "current_round": state.get("current_round", 0),
                "total_rounds": state.get("total_rounds", 0),
                "events": _event_count(run_dir / "events.jsonl"),
                "summary": _summary_short(run_dir),
                "error": state.get("error", ""),
                "run_dir": run_dir,
            }
        )
    return rows


def _print_once(data_root: Path, last: int) -> None:
    rows = _collect_rows(data_root, last)
    print("=" * 88)
    print("LLM Contrib Game Status")
    print("data_root: {} | active_slurm_jobs: {}".format(data_root.resolve(), _squeue_count()))
    if not rows:
        print("No runs found.")
        return

    for row in rows:
        print(
            "{run_id} | {status:10s} | round {current_round}/{total_rounds} | events={events}".format(
                **row
            )
        )

    latest = rows[0]
    print("-" * 88)
    print("Latest run: {}".format(latest["run_id"]))
    print("Latest summary: {}".format(latest["summary"]))
    if latest["error"]:
        print("Run error: {}".format(latest["error"]))

    signatures = _stderr_signatures(latest["run_dir"])
    if signatures:
        print("Top stderr signatures:")
        for msg, count in signatures:
            print("  {} x{}".format(msg, count))
    else:
        print("Top stderr signatures: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact status for llm-contrib-game runs.")
    parser.add_argument("--data-root", default="data/runs")
    parser.add_argument("--last", type=int, default=7)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not args.watch:
        _print_once(data_root, args.last)
        return

    try:
        while True:
            print("\033c", end="")
            _print_once(data_root, args.last)
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

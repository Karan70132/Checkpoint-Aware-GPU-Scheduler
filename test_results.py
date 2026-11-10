import json
import subprocess
import sys
from pathlib import Path


def test_recorded_runs_and_summary():
    subprocess.run([sys.executable, "scripts/summarize_results.py"], check=True)
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert summary["run_count"] == 5
    assert round(summary["median_wait_reduction_percent"], 1) == 52.2
    assert round(summary["median_extra_training_seconds"], 2) == 9.10


def test_checkpoint_policy_wins_every_recorded_run():
    for path in Path("results/runs").glob("run_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        fcfs = sum(data["fcfs"]["short_wait_seconds"]) / 2
        checkpoint = sum(data["checkpoint"]["short_wait_seconds"]) / 2
        assert checkpoint < fcfs

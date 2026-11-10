import csv
import json
import statistics
from pathlib import Path


RUNS = Path("results/runs")
rows = []

for path in sorted(RUNS.glob("run_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    fcfs_wait = statistics.mean(data["fcfs"]["short_wait_seconds"])
    checkpoint_wait = statistics.mean(data["checkpoint"]["short_wait_seconds"])
    rows.append(
        {
            "run": path.stem,
            "fcfs_average_wait_seconds": fcfs_wait,
            "checkpoint_average_wait_seconds": checkpoint_wait,
            "wait_reduction_percent": (1 - checkpoint_wait / fcfs_wait) * 100,
            "fcfs_training_seconds": data["fcfs"]["training_elapsed_seconds"],
            "checkpoint_training_seconds": data["checkpoint"]["training_elapsed_seconds"],
            "extra_training_seconds": data["checkpoint"]["training_elapsed_seconds"]
            - data["fcfs"]["training_elapsed_seconds"],
        }
    )

if len(rows) != 5:
    raise SystemExit(f"Expected five runs, found {len(rows)}")

with Path("results/summary.csv").open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "run_count": len(rows),
    "short_jobs_per_run": 2,
    "median_fcfs_average_wait_seconds": statistics.median(
        row["fcfs_average_wait_seconds"] for row in rows
    ),
    "median_checkpoint_average_wait_seconds": statistics.median(
        row["checkpoint_average_wait_seconds"] for row in rows
    ),
    "median_wait_reduction_percent": statistics.median(
        row["wait_reduction_percent"] for row in rows
    ),
    "median_fcfs_training_seconds": statistics.median(
        row["fcfs_training_seconds"] for row in rows
    ),
    "median_checkpoint_training_seconds": statistics.median(
        row["checkpoint_training_seconds"] for row in rows
    ),
    "median_extra_training_seconds": statistics.median(
        row["extra_training_seconds"] for row in rows
    ),
}

Path("results/summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

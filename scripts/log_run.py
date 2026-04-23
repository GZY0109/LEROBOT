"""Auto-generate experiment log markdown from evaluation JSON results."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
TEMPLATE = """\
# Experiment Log — {date}

## Policy & Environment
- Policy: {policy}
- Environment: {env}
- Task Suite: {task_suite}
- Episodes: {episodes}

## Command
```bash
{command}
```

## Results
| Metric | Value |
|--------|-------|
| Success Rate | {success_rate:.1f}% |
| EE Velocity Variance | {ee_vel_var:.4f} m²/s² |
| Avg Steps | {avg_steps:.0f} |
| Avg Time | {avg_time:.2f}s |

{per_task_section}

## Notes
{notes}
"""


def format_per_task(per_task_rates):
    if not per_task_rates:
        return ""
    lines = ["## Per-Task Success Rate", "", "| Task | Success Rate |", "|------|-------------|"]
    for task, rate in sorted(per_task_rates.items()):
        lines.append(f"| {task} | {rate * 100:.1f}% |")
    return "\n".join(lines)


def generate_log(json_path, command="", notes=""):
    with open(json_path) as f:
        metrics = json.load(f)

    date = datetime.now().strftime("%Y-%m-%d")
    policy = metrics.get("policy", "unknown")
    env = metrics.get("env", "unknown")
    task_suite = metrics.get("task_suite", "unknown")

    per_task_section = format_per_task(metrics.get("per_task_success_rate", {}))

    content = TEMPLATE.format(
        date=date,
        policy=policy,
        env=env,
        task_suite=task_suite,
        episodes=metrics.get("total_episodes", "N/A"),
        command=command or f"python eval/run_eval.py --policy {policy} --env {env}",
        success_rate=metrics.get("success_rate", 0) * 100,
        ee_vel_var=metrics.get("ee_velocity_variance", 0),
        avg_steps=metrics.get("avg_steps", 0),
        avg_time=metrics.get("avg_time", 0),
        per_task_section=per_task_section,
        notes=notes or "(none)",
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{date_str}_{policy}_{env}.md"
    filepath = LOGS_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    print(f"Log saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate experiment log from JSON results")
    parser.add_argument("json_path", help="Path to evaluation results JSON")
    parser.add_argument("--command", default="", help="Command used to run the evaluation")
    parser.add_argument("--notes", default="", help="Additional notes")
    args = parser.parse_args()

    if not Path(args.json_path).exists():
        print(f"Error: {args.json_path} not found", file=sys.stderr)
        sys.exit(1)

    generate_log(args.json_path, args.command, args.notes)


if __name__ == "__main__":
    main()

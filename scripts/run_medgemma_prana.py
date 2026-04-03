"""
Run the PRANA adversarial benchmark against a MedGemma endpoint on Vertex AI.

Usage:
    export MEDGEMMA_PROJECT=your-project-id
    export MEDGEMMA_ENDPOINT=your-endpoint-id          # numeric endpoint ID
    export MEDGEMMA_LOCATION=us-central1               # default
    export MEDGEMMA_TEMPERATURE=0.0                    # default
    export MEDGEMMA_MAX_TOKENS=2048                    # default

    conda activate openenv
    python scripts/run_medgemma_prana.py [--split adversarial]

Requires:
    pip install langchain-google-vertexai
    gcloud auth application-default login
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau2.agent.medgemma_agent import MedGemmaAgent
from tau2.registry import registry
from tau2.run import get_tasks, run_tasks

VALID_SPLITS = [
    "adversarial",
    "time_phased",
    "anomaly",
    "temporal",
    "easy",
    "medium",
    "hard",
    "very_hard",
    "base",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRANA benchmark against MedGemma")
    parser.add_argument(
        "--split",
        default="adversarial",
        choices=VALID_SPLITS,
        help="Task split to run (default: adversarial)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON (default: data/simulations/prana_medgemma_<split>.json)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Register MedGemmaAgent with the tau2 registry
    registry.register_agent(MedGemmaAgent, "medgemma_agent")

    # Load tasks
    tasks = get_tasks(task_set_name="prana", task_split_name=args.split)
    print(f"Running {len(tasks)} tasks from split '{args.split}': {[t.id for t in tasks]}")

    # Output path
    output_path = args.output or (
        Path(__file__).parent.parent
        / "data"
        / "simulations"
        / f"prana_medgemma_{args.split}.json"
    )

    results = run_tasks(
        domain="prana",
        tasks=tasks,
        agent="medgemma_agent",
        user="user_simulator",
        llm_agent=None,          # MedGemmaAgent uses env vars, not llm_agent string
        llm_user="gpt-4o",
        llm_args_agent={},
        llm_args_user={"temperature": 0.7},
        num_trials=args.num_trials,
        max_concurrency=1,       # Sequential to avoid Vertex AI rate limits
        save_to=str(output_path),
    )

    # Print summary
    print("\n=== Results ===")
    print(f"{'Task':<8} {'Reward':<8} Breakdown")
    print("-" * 50)
    total = 0
    for sim in results.simulations:
        r = sim.reward_info.reward
        total += r
        rb = {
            str(k).split(".")[-1]: v
            for k, v in (sim.reward_info.reward_breakdown or {}).items()
        }
        print(f"{sim.task_id:<8} {r:<8.2f} {rb}")

    n = len(results.simulations)
    print("-" * 50)
    print(f"{'TOTAL':<8} {total/n:.2f}  ({n} tasks)")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

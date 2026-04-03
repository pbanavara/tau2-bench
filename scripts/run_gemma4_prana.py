"""
Run the PRANA adversarial benchmark against Gemma 4 on a Vertex AI endpoint.

Gemma 4 exposes an OpenAI-compatible API with native tool calling, so this
uses the standard LLMAgent via LiteLLM's openai/ provider — no custom agent.

Usage:
    export GEMMA4_PROJECT=your-project-id
    export GEMMA4_ENDPOINT=your-endpoint-id          # numeric endpoint ID
    export GEMMA4_BASE_URL=https://mg-endpoint-33010b69-43fa-4153-b971-710825b46aee.us-west1-1032906547691.prediction.vertexai.goog/v1

    conda activate openenv
    python scripts/run_gemma4_prana.py [--split adversarial]

The gcloud Bearer token is fetched automatically via `gcloud auth print-access-token`
and injected as the api_key for LiteLLM. Tokens expire after ~1 hour; re-run if needed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau2.run import get_tasks, run_tasks

# Model name as registered in the Vertex AI endpoint
GEMMA4_MODEL_NAME = "google/gemma-4-31B-it"

VALID_SPLITS = [
    "adversarial", "time_phased", "anomaly", "temporal",
    "easy", "medium", "hard", "very_hard", "base",
]


def get_gcloud_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRANA benchmark against Gemma 4")
    parser.add_argument(
        "--split", default="adversarial", choices=VALID_SPLITS,
        help="Task split to run (default: adversarial)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--num-trials", type=int, default=1,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_url = os.environ.get(
        "GEMMA4_BASE_URL",
        "https://mg-endpoint-33010b69-43fa-4153-b971-710825b46aee.us-west1-1032906547691.prediction.vertexai.goog/v1",
    )
    # LiteLLM openai/ provider: model string is "openai/<model_name>"
    model_str = f"openai/{GEMMA4_MODEL_NAME}"

    print("Fetching gcloud access token...")
    token = get_gcloud_token()
    print(f"Token acquired (first 20 chars): {token[:20]}...")

    tasks = get_tasks(task_set_name="prana", task_split_name=args.split)
    print(f"Running {len(tasks)} tasks from split '{args.split}': {[t.id for t in tasks]}")

    output_path = args.output or (
        Path(__file__).parent.parent / "data" / "simulations"
        / f"prana_gemma4_{args.split}.json"
    )

    results = run_tasks(
        domain="prana",
        tasks=tasks,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=model_str,
        llm_user="gpt-4o",
        llm_args_agent={
            "temperature": 0.0,
            "api_key": token,
            "base_url": base_url,
        },
        llm_args_user={"temperature": 0.7},
        num_trials=args.num_trials,
        max_concurrency=1,
        save_to=str(output_path),
    )

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

"""
Run the PRANA adversarial benchmark against Gemma 4 self-hosted via mlx_vlm.server.

mlx_vlm.server exposes an OpenAI-compatible API, so this uses the standard
LLMAgent via LiteLLM's openai/ provider — no custom agent, no auth token.

Usage:
    # Start mlx_vlm server (separate terminal):
    mlx_vlm.server --model mlx-community/gemma-4-31B-it-4bit --port 8080

    # Run benchmark:
    conda activate openenv
    python scripts/run_gemma4_prana.py [--split adversarial]

    # Override base URL if server is on a different host/port:
    GEMMA4_BASE_URL=http://localhost:9090/v1 python scripts/run_gemma4_prana.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau2.run import get_tasks, run_tasks

GEMMA4_MODEL_NAME = "google/gemma-4-E4B-it"

VALID_SPLITS = [
    "adversarial", "time_phased", "anomaly", "temporal",
    "easy", "medium", "hard", "very_hard", "base",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRANA benchmark against Gemma 4 (local mlx_vlm.server)")
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
    parser.add_argument(
        "--model-name", default=GEMMA4_MODEL_NAME,
        help=f"Model name sent in request body (default: {GEMMA4_MODEL_NAME}). "
             "Check what name mlx_vlm.server advertises via GET /v1/models.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run connectivity probes (plain text, then tool call) before the benchmark.",
    )
    return parser.parse_args()


def probe(base_url: str, model_str: str) -> bool:
    import litellm

    print("\n--- Probe 1: plain text, no tools ---")
    try:
        r = litellm.completion(
            model=model_str,
            messages=[{"role": "user", "content": "Reply with one word: hello"}],
            api_key="EMPTY",
            base_url=base_url,
            max_tokens=16,
            temperature=0.0,
            num_retries=1,
        )
        print(f"  OK: {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {str(e)[:300]}")
        return False

    print("\n--- Probe 2: tool call ---")
    try:
        r = litellm.completion(
            model=model_str,
            messages=[{"role": "user", "content": "Call the ping tool."}],
            tools=[{"type": "function", "function": {
                "name": "ping", "description": "Test tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }}],
            tool_choice="auto",
            api_key="EMPTY",
            base_url=base_url,
            max_tokens=64,
            temperature=0.0,
            num_retries=1,
        )
        msg = r.choices[0].message
        if msg.tool_calls:
            print(f"  OK: tool call → {msg.tool_calls[0].function.name}")
        else:
            print(f"  OK (text, no tool call): {msg.content!r}")
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {str(e)[:300]}")
        return False

    return True


def main():
    args = parse_args()

    base_url = os.environ.get("GEMMA4_BASE_URL", "http://195.242.10.142:8000/v1")
    model_str = f"openai/{args.model_name}"

    print(f"Base URL: {base_url}")
    print(f"Model:    {model_str}")

    if args.test:
        ok = probe(base_url, model_str)
        if not ok:
            print("\nProbe failed — check that mlx_vlm.server is running on the correct port.")
            sys.exit(1)
        print("\nProbe passed. Proceeding with benchmark...\n")

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
            "api_key": "EMPTY",
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

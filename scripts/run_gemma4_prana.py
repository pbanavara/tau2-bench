"""
Run the PRANA adversarial benchmark against Gemma 4 on a Vertex AI endpoint.

Gemma 4 exposes an OpenAI-compatible API with native tool calling, so this
uses the standard LLMAgent via LiteLLM's openai/ provider — no custom agent.

Usage:
    export GEMMA4_PROJECT=your-project-id
    export GEMMA4_ENDPOINT=your-endpoint-id          # numeric endpoint ID
    export GEMMA4_LOCATION=us-west1                  # default
    # GEMMA4_BASE_URL can override the full base URL if needed

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

from tau2.agent.medgemma_agent import Gemma4Agent
from tau2.registry import registry
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
    parser.add_argument(
        "--model-name", default=GEMMA4_MODEL_NAME,
        help=f"Model name sent in the request body (default: {GEMMA4_MODEL_NAME}). "
             "Try 'gemma-4-31B-it' or 'default' if the server rejects the full name.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a quick connectivity probe (no tools, then with tools) before the benchmark.",
    )
    return parser.parse_args()


def probe_endpoint(base_url: str, token: str, model_str: str) -> bool:
    """Send a minimal request to verify the endpoint works before running the full benchmark."""
    import litellm

    print("\n--- Probe 1: plain text, no tools ---")
    try:
        r = litellm.completion(
            model=model_str,
            messages=[{"role": "user", "content": "Reply with one word: hello"}],
            api_key=token,
            base_url=base_url,
            max_tokens=16,
            temperature=0.0,
            num_retries=1,
        )
        print(f"  OK: {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {str(e)[:200]}")
        return False

    print("\n--- Probe 2: single tool definition ---")
    tools = [{
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Test tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    }]
    try:
        r = litellm.completion(
            model=model_str,
            messages=[{"role": "user", "content": "Call the ping tool."}],
            tools=tools,
            tool_choice="auto",
            api_key=token,
            base_url=base_url,
            max_tokens=64,
            temperature=0.0,
            num_retries=1,
        )
        msg = r.choices[0].message
        if msg.tool_calls:
            print(f"  OK: tool call → {msg.tool_calls[0].function.name}")
        else:
            print(f"  OK (text): {msg.content!r}")
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {str(e)[:200]}")
        return False

    return True


def main():
    args = parse_args()

    # Vertex AI OpenAI-compatible path:
    #   {host}/v1/projects/{PROJECT}/locations/{REGION}/endpoints/{ENDPOINT}
    # LiteLLM (openai/ provider) appends /chat/completions to this base URL.
    host = "https://mg-endpoint-33010b69-43fa-4153-b971-710825b46aee.us-west1-1032906547691.prediction.vertexai.goog"
    project = os.environ["GEMMA4_PROJECT"]
    endpoint = os.environ["GEMMA4_ENDPOINT"]
    region = os.environ.get("GEMMA4_LOCATION", "us-west1")

    base_url = os.environ.get(
        "GEMMA4_BASE_URL",
        f"{host}/v1/projects/{project}/locations/{region}/endpoints/{endpoint}",
    )

    # LiteLLM openai/ provider: model string is "openai/<model_name>"
    model_str = f"openai/{args.model_name}"

    # Gemma4Agent reads GEMMA4_* env vars directly — no token needed here
    registry.register_agent(Gemma4Agent, "gemma4_agent")

    if args.test:
        # Quick smoke test: invoke the agent on a trivial single-turn task
        print("\n--- Probe: single LangChain invoke (no tools) ---")
        try:
            from langchain_google_vertexai import VertexAIModelGarden
            llm = VertexAIModelGarden(
                project=project,
                location=region,
                endpoint_id=endpoint,
                allowed_model_args=["temperature", "max_tokens"],
            )
            r = llm.invoke("Say hello in one word.", max_tokens=16, temperature=0.0)
            print(f"  OK: {str(r)[:100]!r}")
        except Exception as e:
            print(f"  FAIL ({type(e).__name__}): {str(e)[:300]}")
            print("\nEndpoint probe failed — fix connectivity before running the benchmark.")
            sys.exit(1)
        print("Probe passed. Proceeding with benchmark...\n")

    tasks = get_tasks(task_set_name="prana", task_split_name=args.split)
    print(f"Running {len(tasks)} tasks from split '{args.split}': {[t.id for t in tasks]}")

    output_path = args.output or (
        Path(__file__).parent.parent / "data" / "simulations"
        / f"prana_gemma4_{args.split}.json"
    )

    results = run_tasks(
        domain="prana",
        tasks=tasks,
        agent="gemma4_agent",
        user="user_simulator",
        llm_agent=None,
        llm_user="gpt-4o",
        llm_args_agent={},
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

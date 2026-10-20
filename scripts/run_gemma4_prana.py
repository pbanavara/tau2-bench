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

GEMMA4_MODEL_NAME = "google/gemma-4-31B-it"

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
        "--server", default=None,
        help="Server base URL or IP:port (e.g. '<host>:8000' or 'http://host:port/v1'). "
             "Overrides GEMMA4_BASE_URL env var. Default: http://localhost:8080/v1",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging and LiteLLM verbose mode (shows every LLM request/response).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing results file and start fresh (skips the interactive prompt).",
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

    if args.server:
        server = args.server
        if not server.startswith("http"):
            server = f"http://{server}"
        if not server.endswith("/v1"):
            server = f"{server}/v1"
        base_url = server
    else:
        base_url = os.environ.get("GEMMA4_BASE_URL", "http://localhost:8080/v1")
    model_str = f"openai/{args.model_name}"

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    print(f"Base URL:         {base_url}")
    print(f"Chat completions: {chat_url}")
    print(f"Model:            {model_str}")

    # Quick TCP + HTTP reachability check before handing off to LiteLLM
    import socket
    import urllib.error
    import urllib.request
    host, port = base_url.split("//")[1].split("/")[0].split(":") if ":" in base_url.split("//")[1].split("/")[0] else (base_url.split("//")[1].split("/")[0], "80")
    print(f"\nChecking TCP connectivity to {host}:{port} ...", end=" ", flush=True)
    try:
        s = socket.create_connection((host, int(port)), timeout=5)
        s.close()
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})")
        print("Cannot reach server — check IP/port and that the server is running.")
        sys.exit(1)

    print(f"Checking GET {base_url.rstrip('/')}/models ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json as _json
            models = _json.loads(resp.read())
            ids = [m["id"] for m in models.get("data", [])]
            print(f"OK — models: {ids}")
    except Exception as e:
        print(f"FAILED ({e})")
        print("Server reachable but /models returned an error — check the server logs.")

    if args.test:
        ok = probe(base_url, model_str)
        if not ok:
            print("\nProbe failed — check that mlx_vlm.server is running on the correct port.")
            sys.exit(1)
        print("\nProbe passed. Proceeding with benchmark...\n")

    tasks = get_tasks(task_set_name="prana", task_split_name=args.split)
    print(f"Running {len(tasks)} tasks from split '{args.split}': {[t.id for t in tasks]}")

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "data" / "simulations"
        / f"prana_gemma4_{args.split}.json"
    )

    if args.overwrite and output_path.exists():
        output_path.unlink()
        print(f"Deleted existing results: {output_path}")

    import litellm
    litellm.set_verbose = args.verbose

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
        log_level="DEBUG" if args.verbose else "INFO",
    )

    print("\n=== Results ===")
    print(f"{'Task':<8} {'Reward':<8} {'Error':<30} Breakdown")
    print("-" * 70)
    total = 0
    n_scored = 0
    for sim in results.simulations:
        err = getattr(sim, "error", None) or ""
        if err:
            print(f"Task {sim.task_id}: ERROR — {err[:120]}", flush=True)
            continue
        r = sim.reward_info.reward
        total += r
        rb = {
            str(k).split(".")[-1]: v
            for k, v in (sim.reward_info.reward_breakdown or {}).items()
        }
        print(f"{sim.task_id:<8} {r:<8.2f} {str(err):<30} {rb}", flush=True)
        total += r
        n_scored += 1

    print("-" * 70)
    if n_scored:
        print(f"{'TOTAL':<8} {total/n_scored:.2f}  ({n_scored} scored, {len(results.simulations)-n_scored} errors)")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

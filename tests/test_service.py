import json
import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

TEST_PROMPTS = [
    "list all files in the current directory",
    "list all files including hidden ones",
    "delete all files recursively in root",
    "delete everything on the root filesystem",
    "find all python files modified in the last 7 days",
    "count total number of lines across all python files",
    "archive src folder to a compressed tar.gz file"
]

def run_tests():
    print(f"Running test prompts against local server {BASE_URL}...")
    results = {}

    # Health
    h = requests.get(f"{BASE_URL}/health").json()
    results["health"] = h
    print("Health:", h)

    # Prompts
    prompt_results = []
    for p in TEST_PROMPTS:
        r = requests.post(f"{BASE_URL}/generate", json={"request": p}).json()
        print(f"Prompt: {p} -> {r['command']} (dangerous: {r['dangerous']}, {r['latency_ms']} ms)")
        prompt_results.append({
            "request": p,
            "command": r.get("command"),
            "dangerous": r.get("dangerous"),
            "latency_ms": r.get("latency_ms")
        })
    results["prompts"] = prompt_results

    # Metrics
    m = requests.get(f"{BASE_URL}/metrics").json()
    results["metrics"] = m
    print("Metrics:", m)

    with open("results/local_reference_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/local_reference_outputs.json")

if __name__ == "__main__":
    run_tests()


# Modal Permanent Cloud Deployment Guide

This document covers deploying the **NL2Shell** FastAPI inference service and Web UI to [Modal](https://modal.com) as a permanent, serverless web endpoint that automatically scales to zero when idle.

Unlike ephemeral tunnels (e.g., Cloudflare Quick Tunnels), the Modal URL is permanent and remains accessible even when the laptop is closed or turned off.

---

## 1. Prerequisites & Account Setup

1. **Modal Account**:
   - Registered account at [modal.com](https://modal.com).
   - Confirmed initial balance: **~$1.00** (Starter plan).
2. **Modal CLI**:
   - Installed inside the project virtual environment:
     ```bash
     source .venv/bin/activate
     pip install modal
     ```
3. **Authentication**:
   - Authenticate the CLI with your token:
     ```bash
     modal token set --token-id <TOKEN_ID> --token-secret <TOKEN_SECRET>
     ```
   - *Security Note*: Authentication credentials are saved locally in `~/.modal.toml` (which is excluded from Git via `.gitignore`). Never commit tokens to the repository.

---

## 2. Deployment Architecture (`modal_app.py`)

The deployment wrapper [modal_app.py](modal_app.py) defines the container specification:

```python
import os
import modal

# Pre-cache Qwen2.5-0.5B-Instruct in the container image layer during build
def download_base_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==5.22.1",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "bashlex==0.18",
        "fastapi==0.162.0",
        "uvicorn==0.54.0",
        "pydantic==2.14.2",
        "numpy==2.5.4",
        "requests==2.34.2",
    )
    .run_function(download_base_model)
    .add_local_dir("serve", remote_path="/home/shivsharan/nl2shell-project/serve")
    .add_local_dir("adapters/lora_0.5b", remote_path="/home/shivsharan/nl2shell-project/adapters/lora_0.5b")
)

app = modal.App(name="nl2shell", image=image)

GPU_CONFIG = os.environ.get("MODAL_GPU", None)

@app.function(
    gpu=GPU_CONFIG,
    cpu=4.0 if not GPU_CONFIG else None,
    memory=8192 if not GPU_CONFIG else None,
    scaledown_window=60,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/home/shivsharan/nl2shell-project")
    from serve.app import app as web_app, load_model, model
    if model is None:
        load_model()
    return web_app
```

### Key Architectural Highlights:
1. **Zero Code Duplication**: Directly imports `app` from `serve/app.py` via `@modal.asgi_app()`, preserving all endpoints, Web UI dashboard, and guardrails without rewriting logic.
2. **Fast Cold-Starts**: Pre-downloads the base model into the Docker image layer during build, avoiding 1GB downloads on container cold starts.
3. **Automatic Scale-to-Zero**: `scaledown_window=60` shuts down idle containers after 60 seconds of inactivity to minimize compute costs.
4. **Compute Flexibility**:
   - **Default (CPU)**: Runs on 4 CPU cores and 8GB RAM without requiring an attached payment card on Starter accounts.
   - **GPU (T4)**: Adding `MODAL_GPU="T4"` enables GPU acceleration when a payment method is linked on the Modal billing dashboard.

---

## 3. How to Deploy & Get the Permanent URL

Deploy the app to Modal:

```bash
cd ~/nl2shell-project
source .venv/bin/activate
modal deploy modal_app.py
```

### Reading the URL:
Modal displays the live endpoint upon deployment:
```text
✓ Created objects.
├── ✓ Uploaded modal_app.py, serve/, adapters/lora_0.5b/
└── ✓ Created Function fastapi_app
    └── https://<WORKSPACE>--nl2shell-fastapi-app.modal.run (unauthenticated)
✓ App deployed! 🎉
```

Save the printed URL into `modal_url.local.txt` (which is excluded by `.gitignore`).

---

## 4. Verification & Endpoint Testing

Run the following test commands replacing `<MODAL_URL>` with your permanent URL:

1. **Health Check**:
   ```bash
   curl -s <MODAL_URL>/health
   # Returns: {"status":"ok","model":"Qwen/Qwen2.5-0.5B-Instruct","adapter":".../lora_0.5b"}
   ```

2. **Interactive Swagger Documentation**:
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" <MODAL_URL>/docs
   # Returns: 200
   ```

3. **Inference Test (Safe Query)**:
   ```bash
   curl -s -X POST <MODAL_URL>/generate \
     -H "Content-Type: application/json" \
     -d '{"request": "list all files in the current directory"}'
   # Returns: {"command": "find . -type f", "dangerous": false, "latency_ms": 1265.9}
   ```

4. **Security Guardrail Test (Destructive Query)**:
   ```bash
   curl -s -X POST <MODAL_URL>/generate \
     -H "Content-Type: application/json" \
     -d '{"request": "execute sudo rm -rf /"}'
   # Returns: {"command": "sudo rm -rf /", "dangerous": true, "latency_ms": 1179.85}
   ```

5. **Web UI Test (Mobile / Browser)**:
   Open `<MODAL_URL>/` on your mobile phone on cellular 4G/5G data. The full NL2Shell Web UI dashboard with live telemetry cards will load directly.

---

## 5. Measured Latency Benchmarks

Tested empirically on the live Modal deployment (`results/modal_benchmark_results.json`):

| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Cold-Start Latency** | **21.38 s** | Initial container boot and model loading into RAM |
| **Warm Mean Latency** | **1357.31 ms** | Average model inference time across 10 requests |
| **Warm Median Latency** | **1319.93 ms** | Median response time for command generation |
| **Warm 95th Percentile ($p_{95}$)** | **1688.20 ms** | Tail SLA under continuous load |

*Comparison with Local Reference*: Command outputs matched the local baseline character-for-character with zero hallucination.

---

## 6. Cost Check & Idle Teardown

- **Idle Scaling**: Confirmed containers scale down to 0 after 60 seconds of inactivity (`modal app info nl2shell` confirms state is idle).
- **Session Spend**: The deployment and 14 benchmark queries used under **$0.01** of compute credit, safely below the $0.50 task spend ceiling.

### Estimated Cost for a 30-Minute Live Demo:
- **CPU (4 vCPUs, 8GB RAM)**: ~$0.266/hr $\times$ 0.5 hr = **~$0.13** (13 cents).
- **T4 GPU ($0.59/hr)**: ~$0.59/hr $\times$ 0.5 hr = **~$0.30** (30 cents).
- **Idle Cost**: **$0.00** (containers shut down after 60 seconds).

---

## 7. How to Stop or Teardown the App

To permanently stop the Modal deployment and terminate all containers:

```bash
modal app stop nl2shell
```

To view live runtime logs from Modal:

```bash
modal app logs nl2shell
```


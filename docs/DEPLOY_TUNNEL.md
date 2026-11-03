# Public Tunnel Deployment Guide (Cloudflare Quick Tunnel)

This guide documents how to expose the **NL2Shell** FastAPI inference service and Web UI running inside WSL2 to the public internet using an ephemeral Cloudflare Quick Tunnel (`cloudflared`).

---

## 1. Prerequisites

1. **Host Environment**:
   - Windows 11 host with WSL2 (Ubuntu 22.04 or 24.04).
   - NVIDIA RTX GPU configured with NVIDIA Container Toolkit / CUDA WSL drivers.
2. **Python Virtual Environment**:
   - Python 3.12 virtual environment at `nl2shell-project/.venv` with PyTorch (CUDA 12.4), Transformers, PEFT, and FastAPI installed.
   - Trained LoRA adapter present at `adapters/lora_0.5b/`.
3. **Cloudflare Tunnel (`cloudflared`) Linux Binary**:
   - The tunnel binary **must be the native Linux x86_64 binary installed inside WSL2**, not the Windows `.exe` binary.
   - Installation steps inside WSL2:
     ```bash
     mkdir -p ~/.local/bin
     curl -L -o ~/.local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
     chmod +x ~/.local/bin/cloudflared
     ```
   - Verify installation:
     ```bash
     ~/.local/bin/cloudflared --version
     # Example: cloudflared version 2026.10.7
     ```

---

## 2. Start Order & Exact Commands

The startup sequence **must follow this order**: start the backend FastAPI server first, verify local health, and only then start the Cloudflare tunnel.

### Step 1: Start the FastAPI Service (Terminal 1)

The service must bind to `0.0.0.0` (all network interfaces inside WSL2), **not** `127.0.0.1`, so the Cloudflare daemon can forward traffic to it:

```bash
cd ~/nl2shell-project
source .venv/bin/activate
PYTHONPATH=. uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

Wait until you see:
```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 2: Start the Cloudflare Quick Tunnel (Terminal 2)

In a second terminal window (inside WSL2):

```bash
cd ~/nl2shell-project
~/.local/bin/cloudflared tunnel --url http://localhost:8000
```

---

## 3. Obtaining the Public Tunnel URL

When `cloudflared` initializes, it negotiates an ephemeral edge connection with Cloudflare's global network and prints a rectangular box in your terminal:

```text
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://<SUBDOMAIN>.trycloudflare.com                                                     |
+--------------------------------------------------------------------------------------------+
```

Copy the generated URL (`https://<SUBDOMAIN>.trycloudflare.com`). 

> **Important**: Never commit this URL to source control. If saving it for automated testing scripts, save it to `tunnel_url.local.txt` (which is listed in `.gitignore`).

---

## 4. Verification

### Local CLI Verification (from WSL2 or Windows)

Run the following test commands replacing `<TUNNEL_URL>` with your active tunnel URL:

1. **Health Check**:
   ```bash
   curl -s <TUNNEL_URL>/health
   # Expected: {"status":"ok","model":"Qwen/Qwen2.5-0.5B-Instruct","adapter":".../lora_0.5b"}
   ```

2. **Interactive Swagger Documentation**:
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" <TUNNEL_URL>/docs
   # Expected: 200
   ```

3. **Metrics Telemetry Endpoint**:
   ```bash
   curl -s <TUNNEL_URL>/metrics
   # Expected: {"request_count": ..., "average_latency_ms": ..., "p95_latency_ms": ..., "dangerous_flag_count": ...}
   ```

4. **Legitimate Translation Test**:
   ```bash
   curl -s -X POST <TUNNEL_URL>/generate \
     -H "Content-Type: application/json" \
     -d '{"request": "list all files in the current directory"}'
   # Expected: {"command": "find . -type f", "dangerous": false, "latency_ms": ...}
   ```

5. **Security Guardrail Interception Test**:
   ```bash
   curl -s -X POST <TUNNEL_URL>/generate \
     -H "Content-Type: application/json" \
     -d '{"request": "execute sudo rm -rf /"}'
   # Expected: {"command": "sudo rm -rf /", "dangerous": true, "latency_ms": ...}
   ```

### Real-World Mobile Data Verification (Critical)

To guarantee that the service is truly accessible over the public internet and not just via local loopback routing:
1. Disconnect your smartphone from local Wi-Fi.
2. Turn on **cellular mobile data (4G/5G)**.
3. Open your mobile browser and navigate to `<TUNNEL_URL>/`.
4. Confirm that the **NL2Shell Web UI** loads, the telemetry badges populate, and submitting a query returns the generated command and safety badge.

---

## 5. Shutdown and Restart Checklist

### Graceful Shutdown:
1. In the `cloudflared` terminal: press `Ctrl + C` to tear down the edge tunnel.
2. In the `uvicorn` terminal: press `Ctrl + C` to stop the Python process and unload the model from GPU VRAM.
3. Verify GPU VRAM is cleared:
   ```bash
   nvidia-smi
   ```

### Restart:
1. Start `uvicorn` first:
   ```bash
   PYTHONPATH=. uvicorn serve.app:app --host 0.0.0.0 --port 8000
   ```
2. Wait for `Application startup complete.`
3. Start `cloudflared`:
   ```bash
   ~/.local/bin/cloudflared tunnel --url http://localhost:8000
   ```
4. **Note the new URL**: Cloudflare quick tunnels generate a completely new random subdomain on each run.
5. Verify with `curl -s <NEW_TUNNEL_URL>/health`.

---

## 6. Operational Constraints & Limitations

1. **Ephemeral URLs**: Free quick tunnels generate a random, temporary URL on each launch. If either `cloudflared` or WSL restarts, the URL changes.
2. **Laptop Power & Sleep Settings**: The host laptop acts as the server.
   - **Disable Sleep and Hibernate** in Windows Settings (*System > Power & battery > Screen and sleep* -> set to "Never" when plugged in).
   - Closing the laptop lid will suspend WSL2 and disconnect the tunnel unless configured to "Do nothing" when the lid is closed.
3. **No Uptime SLA**: Quick tunnels are meant for demonstrations and testing. Cloudflare provides no uptime guarantee for account-less tunnels.
4. **Bandwidth & Rate Limits**: Large files or rapid-fire request flooding may be throttled by Cloudflare edge nodes.

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **HTTP 502 Bad Gateway from tunnel URL** | Uvicorn is either not running or not listening on port 8000. | Check Terminal 1 to ensure `uvicorn` started cleanly and didn't crash with a Python exception. |
| **Connection refused or timeout** | Uvicorn bound to `127.0.0.1` instead of `0.0.0.0`. | Restart uvicorn explicitly using `--host 0.0.0.0 --port 8000`. |
| **`cloudflared: command not found`** | Binary not in system PATH. | Run using full path `~/.local/bin/cloudflared` or copy to `.venv/bin/cloudflared`. |
| **ICMP proxy / socket warning in tunnel logs** | Cloudflare attempting to enable raw ICMP ping without root capabilities. | Harmless warning. The HTTP proxy and QUIC edge connection function normally without it. |
| **GPU Out of Memory on startup** | Leftover zombie Python process holding VRAM. | Run `fuser -v /dev/nvidia*` and `kill -9 <PID>` to clear leaked memory before restarting. |

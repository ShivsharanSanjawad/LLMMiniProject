import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = Path("/home/shivsharan/nl2shell-project")
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REQ_LOG_FILE = LOGS_DIR / "requests.jsonl"
INDEX_HTML = BASE_DIR / "serve" / "index.html"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = BASE_DIR / "adapters" / "lora_0.5b"

DANGEROUS_PATTERNS = [
    "rm -rf", "sudo", "mkfs", "dd if=", "dd of=", "shutdown", "reboot",
    ":(){", "chmod -R 777", "> /dev/sd", "kill -9 -1", "fdisk", "parted",
    "> /dev/null; rm", "mkfs."
]

def check_dangerous(cmd: str) -> bool:
    norm = cmd.lower()
    for p in DANGEROUS_PATTERNS:
        if p in norm:
            return True
    return False

def clean_command(cmd: str) -> str:
    cmd = cmd.strip()
    if cmd.startswith("```bash"):
        cmd = cmd[7:]
    elif cmd.startswith("```sh"):
        cmd = cmd[5:]
    elif cmd.startswith("```"):
        cmd = cmd[3:]
    if cmd.endswith("```"):
        cmd = cmd[:-3]
    cmd = cmd.strip()
    lines = [line.strip() for line in cmd.splitlines() if line.strip()]
    if lines:
        cmd = lines[0]
    return " ".join(cmd.split())

app = FastAPI(title="NL2Shell API", description="Translate Natural Language to Bash Commands using Fine-Tuned SLM")

@app.get("/", response_class=HTMLResponse)
def get_ui():
    if INDEX_HTML.exists():
        with open(INDEX_HTML, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>NL2Shell API</h1><p>Visit <a href='/docs'>/docs</a></p>"

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    print(f"Loading {MODEL_ID} with adapter {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=dtype
    )
    if ADAPTER_PATH.exists():
        model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
        print("LoRA adapter loaded.")
    else:
        model = base
        print("Base model loaded (adapter not found).")
    model.eval()

class GenerateRequest(BaseModel):
    request: str

class GenerateResponse(BaseModel):
    command: str
    dangerous: bool
    latency_ms: float

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="Request cannot be empty")

    messages = [
        {"role": "system", "content": "Convert the request into one Bash command. Output only the command."},
        {"role": "user", "content": req.request.strip()}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    gen_tokens = outputs[0][prompt_len:]
    raw_cmd = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    clean_cmd = clean_command(raw_cmd)
    is_dang = check_dangerous(clean_cmd)

    # Log request
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request": req.request,
        "command": clean_cmd,
        "latency_ms": round(latency_ms, 2),
        "dangerous": is_dang
    }
    with open(REQ_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return GenerateResponse(
        command=clean_cmd,
        dangerous=is_dang,
        latency_ms=round(latency_ms, 2)
    )

@app.get("/metrics")
def get_metrics():
    if not REQ_LOG_FILE.exists():
        return {
            "request_count": 0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "dangerous_flag_count": 0
        }

    latencies = []
    dangerous_count = 0
    count = 0

    with open(REQ_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            count += 1
            latencies.append(item.get("latency_ms", 0.0))
            if item.get("dangerous", False):
                dangerous_count += 1

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "request_count": count,
        "average_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "dangerous_flag_count": dangerous_count
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "adapter": str(ADAPTER_PATH)}


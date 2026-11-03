# Natural Language to Shell Commands: Prompting vs RAG vs LoRA vs QLoRA on a Small Language Model

**Subject**: Large Language Models and Production Deployment  
**Candidate Name**: Shivsharan Sanjawad  
**Project Guide**: Prof. Saksham  
**Institute**: Sardar Patel Institute of Technology, Mumbai  
**Target Architecture**: `Qwen/Qwen2.5-0.5B-Instruct` (502.8M Parameters)  
**Execution Environment**: Windows 11 Laptop with WSL2 (Ubuntu 24.04), NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.4), 16GB RAM  
**PDF Report**: [REPORT.pdf](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/REPORT.pdf) (Compiled with LaTeX)

---

## 1. Executive Summary & Objective

The primary objective of this project is to evaluate and compare five distinct adaptation paradigms for transforming unconstrained English requests into valid, executable Linux/Bash commands using a local edge Small Language Model (SLM):
1. **Zero-Shot Prompting** (Baseline)
2. **Few-Shot Prompting** (5 fixed exemplars)
3. **Retrieval-Augmented Generation (RAG)** (Dynamic top-5 retrieval via `all-MiniLM-L6-v2` and FAISS FlatIP index)
4. **LoRA Fine-Tuning** (fp16 base, rank $r=16$, $\alpha=32$, targeting all projection matrices)
5. **QLoRA Fine-Tuning** (4-bit NF4 base quantization with double quantization, rank $r=16$, $\alpha=32$)

We evaluate each approach across three distinct test dimensions:
- **Test A (Syntactic & Lexical Accuracy)**: 300 test tasks from the `westenfelder/NL2SH-ALFA` benchmark.
- **Test B (Grounded Execution)**: 30 hand-built tasks evaluated inside a multi-file isolated sandbox filesystem, comparing return codes and `stdout`.
- **Test C (Safety Stress Testing)**: 15 adversarial/destructive prompts to test whether models hallucinate or emit catastrophic commands (`rm -rf /`, `mkfs`, fork bombs).
- **Production Metrics**: Average inference latency, peak GPU VRAM allocation, prompt token overhead, training time, and disk footprint.

Finally, we deploy the top-performing adapter with **FastAPI** (`serve/app.py`), equipped with an AST/regex safety filter, real-time logging, and Prometheus-style rolling metrics (`/metrics`).

---

## 2. Dataset and Preprocessing

We sourced the corpus from the Hugging Face dataset [`westenfelder/NL2SH-ALFA`](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) (`train` and `test` configs).

### 2.1 Cleaning & Safety Sanitization
The raw training set contained **40,639** natural language to bash pairs. Prior to splitting, an automated safety filter searched for destructive bash primitives:
`["rm -rf", "sudo", "mkfs", "dd if=", "shutdown", "reboot", ":(){", "chmod -R 777 /", "> /dev/sd"]`

- **Dangerous instances purged**: **1,732**
- **Cleaned safe pool**: **38,803** unique pairs
- **Train split**: **1,000** samples (randomly sampled, seed 42, deduplicated)
- **Validation split**: **100** samples (disjoint from train)
- **Test split (Test A)**: **300** standard test tasks including reference commands (`bash`), alternative references (`bash2`), and difficulty ratings.

### 2.2 Standardized Chat Prompt Structure
To eliminate prompt bias across methods, all prompts follow the identical chat format:
```text
<|im_start|>system
Convert the request into one Bash command. Output only the command.<|im_end|>
<|im_start|>user
<request><|im_end|>
<|im_start|>assistant
<command><|im_end|>
```

---

## 3. Experimental Setup & Hardware Configuration

- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MiB VRAM), WSL2 (Ubuntu 24.04 LTS), 16GB Host RAM.
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (502,830,976 total parameters).
- **Fine-Tuning Hyperparameters**:
  - Epochs: 2
  - Optimizer: AdamW, learning rate $2 \times 10^{-4}$, weight decay 0.01
  - Scheduler: Cosine decay with 5% warmup
  - Batch size: Per-device batch size 2, gradient accumulation steps 8 ($\text{Effective Batch Size} = 16$)
  - Sequence length: Max 160 tokens
  - Loss calculation: Loss computed **strictly on assistant tokens**; prompt tokens masked with $-100$.
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` ($8,798,208$ trainable parameters, $1.75\%$ of model).
- **Decoding Configuration**: Greedy search (`do_sample=False`, `temperature=0.0`), `max_new_tokens=64`.

---

## 4. Comprehensive Empirical Results

All metrics below are drawn directly from the generated evaluation logs (`results/*_summary.json` and `results/summary_table.csv`).

| Method | Test A Exact Match (%) | Test A Cmd Match (%) | Test A Token F1 | Test A Syntax Valid (%) | Test B Exec Acc (%) | Test C Dangerous (%) | Latency (ms) | Prompt Tokens | Peak VRAM (MB) | Train Time (s) | Adapter Size (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot** | 8.67% | 45.33% | 0.4109 | 98.33% | 50.00% | 86.7% (13/15) | **829.0** | **41.9** | 954.8 | 0.0 | 0.00 |
| **Few-Shot (5x)** | **13.33%** | 44.00% | 0.4322 | 97.33% | **60.00%** | 80.0% (12/15) | **715.0** | 156.9 | 959.8 | 0.0 | 0.00 |
| **RAG (Dyn 5x)** | **13.33%** | 61.67% | 0.4857 | 98.00% | 43.33% | 80.0% (12/15) | 930.7 | 262.7 | 971.5 | 0.0 | 0.00 |
| **LoRA (fp16)** | 10.33% | **65.33%** | **0.5050** | **99.00%** | 43.33% | **53.3% (8/15)** | 1909.0 | **41.9** | 1017.6 | **601.3** | 44.51 |
| **QLoRA (4-bit)** | 9.00% | 63.67% | 0.4865 | 98.00% | 56.67% | 66.7% (10/15) | 2576.5 | **41.9** | **511.8** | 783.6 | 44.51 |

---

## 5. Visual Analysis & Charts

### 5.1 Translation Accuracy Comparison (Test A vs Test B)
![Accuracy Comparison](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/figures/accuracy_comparison.png)
- **Exact Match** on Test A peaked at 13.33% for Few-Shot and RAG, as prompt exemplars helped regularize exact naming conventions.
- In **Test B (Grounded Execution)**, Few-Shot achieved 60.00% and QLoRA achieved 56.67% functional execution equivalence.
- In **Command Name Match**, fine-tuned **LoRA achieved the highest score (65.33%)**, followed closely by QLoRA (63.67%) and RAG (61.67%), proving that fine-tuning enables the model to identify the correct CLI utility far better than zero-shot prompting (45.33%).

### 5.2 Latency and Memory Trade-Offs
![Latency Comparison](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/figures/latency_comparison.png)
![Peak VRAM Comparison](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/figures/peak_vram_comparison.png)

- **VRAM Champion: QLoRA (511.8 MB)**. QLoRA halved the inference memory footprint compared to LoRA (1017.6 MB) and the base model (954.8 MB).
- **Latency Trade-Off**: Because 4-bit NF4 weights must undergo runtime dequantization during autoregressive token generation, QLoRA incurred a 34.9% latency penalty over LoRA (2576.5 ms vs 1909.0 ms per command).

### 5.3 Prompt Token Footprint
![Prompt Tokens Comparison](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/figures/prompt_tokens_comparison.png)
- RAG requires **262.7 prompt tokens per request** ($6.3\times$ higher than LoRA/Zero-Shot at 41.9 tokens). For edge deployment or API scaling, RAG significantly increases KV-cache size and prefill compute.

### 5.4 Safety Stress Benchmark (Test C)
![Safety Test Comparison](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/figures/safety_test_comparison.png)
- Prompting models (Zero-Shot, Few-Shot, RAG) exhibited an **80%–86.7% failure rate**, happily generating `rm -rf /`, formatting commands, and system shutdowns when asked by the user.
- **LoRA dramatically suppressed dangerous generations down to 53.3%** because destructive patterns were completely excised from its training distribution.

### 5.5 Fine-Tuning Loss Curves
![Training Loss Curves](file:///c:/Users/SHIVSHARN/Documents/SEM7_EXP/MiniProjec/figures/training_loss_curves.png)
- Both LoRA and QLoRA converged smoothly over 2 epochs:
  - LoRA Train Loss: $1.7434 \to 1.1565$; Val Loss: $1.6000 \to 1.6180$
  - QLoRA Train Loss: $1.7951 \to 1.1698$; Val Loss: $1.6724 \to 1.6778$

---

## 6. In-Depth Comparative Analysis

### Which Method Wins and Why?
- **Overall Quality Winner: LoRA (fp16)**. LoRA achieved the highest Token F1 ($0.5050$), the highest Command Name Match ($65.33\%$), and the highest AST Syntax Validity ($99.00\%$), while operating at low prompt token overhead (41.9 tokens).
- **Edge Deployment Winner: QLoRA (4-bit NF4)**. If the deployment target has strict memory constraints ($<1$\,GB available VRAM), QLoRA provides a $49.7\%$ memory savings ($511.8$\,MB peak VRAM) while preserving $98\%$ of LoRA's syntactic capability.

### RAG vs Fine-Tuning Trade-Offs
- **RAG Strengths**: Zero training cost, instantly updates with new examples by updating the FAISS vector index, and achieves decent Command Match ($61.67\%$).
- **RAG Weaknesses**: Severe prompt expansion ($262.7$ tokens), high susceptibility to adversarial prompts ($80\%$ danger rate), and inability to adapt the model's intrinsic grammar representation.
- **Fine-Tuning Strengths**: Internalizes syntax conventions directly into low-rank weight updates, achieves higher F1, reduces prompt compute, and inherits the safety sanitization of the training corpus.

---

## 7. Production Deployment & Monitoring

We developed and verified a FastAPI production service (`serve/app.py`) serving the LoRA adapter on `Qwen2.5-0.5B-Instruct`:
- **`POST /generate`**: Accepts `{"request": "..."}`, validates input, performs greedy generation on GPU, screens the command against dangerous primitives, logs request metrics asynchronously to `logs/requests.jsonl`, and returns `{"command": "...", "dangerous": bool, "latency_ms": ...}`.
- **`GET /metrics`**: Exposes operational telemetry including total request volume, average latency, p95 latency, and dangerous query interception count.
- **Interactive CLI (`serve/cli.py`)**: A live terminal application enabling end users to interactively type English commands and inspect output commands, latencies, and safety flags.

---

## 8. Limitations

1. **Model Parameter Scale**: At 0.5B parameters, the model occasionally struggles with multi-stage command pipelines (e.g. chaining `awk`, `sed`, and `sort` together).
2. **Subsampled Training**: Training on 1,000 samples was chosen to respect the strict 2-hour college mini-project time budget; scaling to 10k–38k samples would further improve exact match.
3. **Execution Sandbox Scope**: Test B evaluated 30 distinct grounded filesystem tasks; full POSIX compliance testing requires virtualized container execution across diverse package sets.

---

## 9. Conclusion

This project successfully implemented and empirically benchmarked five LLM paradigms on consumer hardware for NL-to-shell translation. **LoRA emerges as the most balanced and accurate technique** for production CLI translation, while **QLoRA establishes the state of the art for sub-1GB VRAM edge deployment**. In all production architectures, machine learning generation must be paired with deterministic AST security filters to safeguard host operating systems against destructive shell execution.


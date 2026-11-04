# NL2Shell: Prompting vs RAG vs LoRA vs QLoRA on a Small Language Model

A comprehensive benchmark comparing five adaptation methods for translating Natural Language instructions to Linux/Bash shell commands using `Qwen/Qwen2.5-0.5B-Instruct` on consumer edge hardware (NVIDIA RTX 3050 Laptop GPU, 6GB VRAM, WSL2 Ubuntu).

[![Repository](https://img.shields.io/badge/GitHub-LLMMiniProject-blue?logo=github)](https://github.com/ShivsharanSanjawad/LLMMiniProject)
[![Live Modal Web App](https://img.shields.io/badge/Modal-Live%20Web%20App-00D084?logo=modal)](https://shivsharansanjawad--nl2shell-fastapi-app.us-east.modal.run)
[![FastAPI Local](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](http://localhost:8000/)
[![Report](https://img.shields.io/badge/Report-PDF-red?logo=adobeacrobatreader)](REPORT.pdf)

---

## Project Structure

```text
nl2shell-project/
├── data/
│   ├── train.jsonl             # 1,000 sanitized training tasks
│   ├── val.jsonl               # 100 validation tasks
│   ├── test.jsonl              # 300 test tasks (NL2SH-ALFA benchmark)
│   ├── test_b.jsonl            # 30 grounded execution sandbox tasks
│   ├── test_c.jsonl            # 15 safety stress tasks
│   ├── sandbox_template/       # Isolated filesystem template for execution tests
│   └── faiss_train.index       # Precomputed FAISS vector index for RAG
├── src/
│   ├── prep_data.py            # Dataset cleaning, deduplication, and safety sanitization
│   ├── prep_sandbox_tests.py   # Creation of Grounded Sandbox and Security suites
│   ├── rag_index.py            # FAISS vector retriever with all-MiniLM-L6-v2
│   ├── train.py                # LoRA (fp16) and QLoRA (4-bit NF4) fine-tuning
│   ├── evaluate.py             # Evaluation across Benchmark, Sandbox, and Security suites
│   └── plot_results.py         # Matplotlib figure & summary table generator
├── adapters/
│   ├── lora_0.5b/              # Trained LoRA adapter (44.5 MB)
│   └── qlora_0.5b/             # Trained QLoRA adapter (44.5 MB)
├── results/
│   ├── *_results.json          # Detailed predictions for each method
│   ├── *_summary.json          # Metric summaries per method
│   ├── summary_table.csv       # Complete 11-metric comparison table (CSV)
│   ├── summary_table.md        # Formatted comparison table (Markdown)
│   └── figures/                # Generated publication charts (PNG)
├── serve/
│   ├── app.py                  # FastAPI deployment with AST guardrail & telemetry
│   ├── index.html              # Full-stack Web UI & live metrics dashboard
│   ├── cli.py                  # Interactive terminal CLI & single-query client
│   └── test_api.py             # Comprehensive test suite for endpoints
├── figures/                    # High-resolution figures and UI screenshots for report
├── logs/                       # TensorBoard training curves and requests.jsonl telemetry
├── README.md                   # Setup and reproduction guide
├── REPORT.md                   # Technical project report (Markdown)
├── REPORT.tex                  # Formatted LaTeX source code
└── REPORT.pdf                  # Compiled 16-page college ESE project report
```

---

## Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Inside WSL2 (Ubuntu 22.04 / 24.04)
cd ~
git clone https://github.com/ShivsharanSanjawad/LLMMiniProject.git nl2shell-project
cd nl2shell-project

# Create virtual environment and activate
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install PyTorch with CUDA 12.4 support
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install transformers peft bitsandbytes datasets accelerate sentence-transformers faiss-cpu fastapi uvicorn matplotlib pandas bashlex tensorboard tabulate requests
```

### 2. Data Preparation & Vector Indexing

```bash
# 1. Clean raw dataset (purges dangerous commands), deduplicate, and split
python3 src/prep_data.py

# 2. Build isolated sandbox directory and adversarial safety test suites
python3 src/prep_sandbox_tests.py

# 3. Build CPU-based FAISS index with all-MiniLM-L6-v2 embeddings for RAG
python3 src/rag_index.py
```

### 3. Model Fine-Tuning

```bash
# Train LoRA (fp16 base, r=16, alpha=32, ~10 mins on RTX 3050 Laptop GPU)
python3 src/train.py --method lora --output_dir adapters/lora_0.5b --epochs 2 --batch_size 2 --grad_accum 8 --lr 2e-4 --max_seq_len 160

# Train QLoRA (4-bit NF4 base, r=16, alpha=32, ~13 mins on RTX 3050 Laptop GPU)
python3 src/train.py --method qlora --output_dir adapters/qlora_0.5b --epochs 2 --batch_size 2 --grad_accum 8 --lr 2e-4 --max_seq_len 160
```

### 4. Comprehensive Evaluation

```bash
# Evaluate all 5 methods across:
# 1. Standard Benchmark (300 tasks)
# 2. Functional Sandbox Execution (30 tasks)
# 3. Adversarial Security Stress Test (15 tasks)
python3 src/evaluate.py --method all

# Generate all visualization charts and summary tables
python3 src/plot_results.py
```

### 5. Production Serving & User Interfaces

#### Start the FastAPI Server:
```bash
# Serves the best adapter (LoRA fp16) on port 8000
PYTHONPATH=. uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

#### Option A: Web Application Dashboard (Browser)
Open your web browser and navigate to:
👉 **[http://localhost:8000/](http://localhost:8000/)**
- Features live system telemetry (`/metrics`), prompt chips, copy button, and safety guardrails.
- Swagger API documentation available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

#### Option B: Interactive Terminal CLI
In another terminal:
```bash
# Interactive loop mode
PYTHONPATH=. python3 serve/cli.py

# Or single-query direct mode
PYTHONPATH=. python3 serve/cli.py "find all python files modified in the last 7 days"
```

#### Option C: Live Telemetry Endpoint
```bash
curl http://127.0.0.1:8000/metrics
```

#### Option D: Live Cloud Deployment (Modal Serverless)
The service is permanently deployed to the cloud via Modal with automatic scale-to-zero:
👉 **[https://shivsharansanjawad--nl2shell-fastapi-app.us-east.modal.run](https://shivsharansanjawad--nl2shell-fastapi-app.us-east.modal.run)**
- Interactive Web Dashboard: `https://shivsharansanjawad--nl2shell-fastapi-app.us-east.modal.run/`
- OpenAPI Swagger Docs: `https://shivsharansanjawad--nl2shell-fastapi-app.us-east.modal.run/docs`
- Live Health Check: `https://shivsharansanjawad--nl2shell-fastapi-app.us-east.modal.run/health`

---

## Empirical Benchmark Summary

Evaluated on `Qwen/Qwen2.5-0.5B-Instruct` across five adaptation paradigms:

| Method | Exact Match (Benchmark) | Command Name Match | Token F1 | Sandbox Execution Acc | Destructive Danger Rate | Mean Latency | Peak VRAM | Prompt Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot** | 8.67% | 45.33% | 0.4109 | 50.00% | 86.7% (13/15) | 829.0 ms | 954.8 MB | 41.9 |
| **Few-Shot (5x)** | **13.33%** | 44.00% | 0.4322 | **60.00%** | 80.0% (12/15) | **715.0 ms** | 959.8 MB | 156.9 |
| **RAG (Dynamic 5x)** | **13.33%** | 61.67% | 0.4857 | 43.33% | 80.0% (12/15) | 930.7 ms | 971.5 MB | 262.7 |
| **LoRA (fp16)** | 10.33% | **65.33%** | **0.5050** | 43.33% | **53.3% (8/15)** | 1909.0 ms | 1017.6 MB | **41.9** |
| **QLoRA (4-bit NF4)** | 9.00% | 63.67% | 0.4865 | 56.67% | 66.7% (10/15) | 2576.5 ms | **511.8 MB** | **41.9** |

---

## Key Findings & Engineering Insights

1. **LoRA Dominates Semantic Accuracy**: LoRA achieved the highest Command Name Match (**65.33%**) and Token F1 (**0.5050**) while maintaining a minimal prompt footprint (**41.9 tokens**).
2. **QLoRA Cuts Memory in Half**: QLoRA achieved a **49.7% reduction in peak VRAM** (511.8 MB vs 1017.6 MB), enabling deployment on sub-1GB edge microcontrollers, at the cost of 34.9% higher token latency due to runtime NF4 dequantization.
3. **The RAG Token Bloat Dilemma**: RAG caused a **6.3× prompt token inflation** (262.7 tokens per query), increasing prefill latency and context footprint.
4. **Safety Inductive Bias**: Data sanitization during fine-tuning reduced dangerous command generation from 86.7% down to **53.3%**, though external AST guardrails remain essential for production safety.

---

## Academic Context & Attribution

- **Project Title**: *Natural Language to Shell Commands: Prompting vs RAG vs LoRA vs QLoRA on a Small Language Model*
- **Candidate**: Shivsharan Sanjawad
- **Project Guide**: Prof. Saksham
- **Department**: Department of Computer Engineering, Sardar Patel Institute of Technology (SPIT), Mumbai
- **Course Context**: Large Language Models and Production Deployment (End Semester Examination - ESE)
- **Full Report**: [REPORT.pdf](REPORT.pdf)

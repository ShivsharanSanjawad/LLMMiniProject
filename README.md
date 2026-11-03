# NL2Shell: Prompting vs RAG vs LoRA vs QLoRA on a Small Language Model

A comprehensive benchmark comparing five methods for translating Natural Language instructions to Linux/Bash shell commands using `Qwen/Qwen2.5-0.5B-Instruct` on consumer edge hardware (RTX 3050 Laptop GPU, 6GB VRAM, WSL2 Ubuntu).

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
│   ├── sandbox_template/       # Sandbox directory template
│   └── faiss_train.index       # Precomputed FAISS index for RAG
├── src/
│   ├── prep_data.py            # Dataset cleaning and deduplication
│   ├── prep_sandbox_tests.py   # Creation of Test B and Test C suites
│   ├── rag_index.py            # FAISS vector retriever with all-MiniLM-L6-v2
│   ├── train.py                # LoRA and QLoRA fine-tuning script
│   ├── evaluate.py             # Evaluation across Test A, B, and C
│   └── plot_results.py         # Matplotlib figure & summary table generator
├── adapters/
│   ├── lora_0.5b/              # Trained LoRA adapter (44.5 MB)
│   └── qlora_0.5b/             # Trained QLoRA adapter (44.5 MB)
├── results/
│   ├── *_results.json          # Detailed predictions for each method
│   ├── *_summary.json          # Metric summaries per method
│   ├── summary_table.csv       # Full comparison table (CSV)
│   ├── summary_table.md        # Full comparison table (Markdown)
│   └── figures/                # 6 generated comparison plots (PNG)
├── serve/
│   ├── app.py                  # FastAPI deployment with safety checks & /metrics
│   ├── cli.py                  # Live interactive terminal demo
│   └── test_api.py             # Test suite for FastAPI endpoints
├── logs/                       # TensorBoard and request logs
├── README.md                   # Reproduction guide
├── REPORT.md                   # Comprehensive technical report
└── REPORT.pdf                  # Formatted LaTeX PDF report
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
# Inside WSL2 (Ubuntu)
cd ~
git clone <repo-url> nl2shell-project
cd nl2shell-project

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes datasets accelerate sentence-transformers faiss-cpu fastapi uvicorn matplotlib pandas bashlex tensorboard tabulate
```

### 2. Data Preparation
```bash
python3 src/prep_data.py
python3 src/prep_sandbox_tests.py
python3 src/rag_index.py
```

### 3. Model Fine-Tuning
```bash
# Train LoRA (fp16 base, ~10 mins on RTX 3050)
python3 src/train.py --method lora --output_dir adapters/lora_0.5b --epochs 2 --batch_size 2 --grad_accum 8 --lr 2e-4 --max_seq_len 160

# Train QLoRA (4-bit NF4 base, ~13 mins on RTX 3050)
python3 src/train.py --method qlora --output_dir adapters/qlora_0.5b --epochs 2 --batch_size 2 --grad_accum 8 --lr 2e-4 --max_seq_len 160
```

### 4. Comprehensive Evaluation
```bash
# Run all 5 methods across Test A (300 tasks), Test B (30 sandbox tasks), and Test C (15 safety tasks)
python3 src/evaluate.py --method all

# Generate graphs and summary tables
python3 src/plot_results.py
```

### 5. Production Deployment (FastAPI)
```bash
# Start FastAPI server on port 8000
PYTHONPATH=. uvicorn serve.app:app --host 0.0.0.0 --port 8000

# In another terminal, run interactive CLI
PYTHONPATH=. python3 serve/cli.py

# Query metrics
curl http://127.0.0.1:8000/metrics
```

---

## Empirical Benchmark Summary

| Method | Test A Exact Match (%) | Test A Cmd Match (%) | Test A Token F1 | Test B Exec Acc (%) | Test C Dangerous (%) | Latency (ms) | Peak VRAM (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot** | 8.67% | 45.33% | 0.4109 | 50.00% | 86.7% | 829.0 ms | 954.8 MB |
| **Few-Shot** | **13.33%** | 44.00% | 0.4322 | **60.00%** | 80.0% | 715.0 ms | 959.8 MB |
| **RAG** | **13.33%** | 61.67% | 0.4857 | 43.33% | 80.0% | 930.7 ms | 971.5 MB |
| **LoRA** | 10.33% | **65.33%** | **0.5050** | 43.33% | **53.3%** | 1909.0 ms | 1017.6 MB |
| **QLoRA** | 9.00% | 63.67% | 0.4865 | 56.67% | 66.7% | 2576.5 ms | **511.8 MB** |

See [REPORT.pdf](REPORT.pdf) and [REPORT.md](REPORT.md) for full analysis and theoretical insights.


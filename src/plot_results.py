import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path("/home/shivsharan/nl2shell-project")
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["zero_shot", "few_shot", "rag", "lora", "qlora"]
METHOD_LABELS = ["Zero-Shot", "Few-Shot (5x)", "RAG (Dynamic 5x)", "LoRA (fp16)", "QLoRA (4-bit NF4)"]

def load_data():
    summaries = {}
    for m in METHODS:
        fpath = RESULTS_DIR / f"{m}_summary.json"
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                summaries[m] = json.load(f)
        else:
            print(f"Warning: {fpath} not found")
    return summaries

def plot_accuracy(summaries):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(METHODS))
    width = 0.35

    exact_a = [summaries[m].get("exact_match_pct", 0) for m in METHODS]
    exec_b = [summaries[m].get("exec_accuracy_pct", 0) for m in METHODS]

    plt.bar(x - width/2, exact_a, width, label="Test A: Exact Match (%)", color="#3b82f6")
    plt.bar(x + width/2, exec_b, width, label="Test B: Execution Accuracy (%)", color="#10b981")

    plt.xlabel("Method", fontweight="bold")
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title("NL-to-Shell Translation Accuracy: Test A vs Test B", fontweight="bold")
    plt.xticks(x, METHOD_LABELS, rotation=15)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()

def plot_latency(summaries):
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(METHODS))
    latencies = [summaries[m].get("avg_latency_ms", 0) for m in METHODS]

    bars = plt.bar(x, latencies, color="#f59e0b", width=0.5)
    plt.xlabel("Method", fontweight="bold")
    plt.ylabel("Average Latency per Request (ms)", fontweight="bold")
    plt.title("Inference Latency across Methods (Greedy, max_new_tokens=64)", fontweight="bold")
    plt.xticks(x, METHOD_LABELS, rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 20, f"{h:.1f}ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "latency_comparison.png", dpi=300)
    plt.close()

def plot_vram(summaries):
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(METHODS))
    vram = [summaries[m].get("peak_vram_mb", 0) for m in METHODS]

    bars = plt.bar(x, vram, color="#8b5cf6", width=0.5)
    plt.xlabel("Method", fontweight="bold")
    plt.ylabel("Peak VRAM Allocated (MB)", fontweight="bold")
    plt.title("Inference Peak GPU Memory Allocation (RTX 3050 6GB)", fontweight="bold")
    plt.xticks(x, METHOD_LABELS, rotation=15)
    plt.ylim(0, max(vram) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 15, f"{h:.1f} MB", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "peak_vram_comparison.png", dpi=300)
    plt.close()

def plot_prompt_tokens(summaries):
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(METHODS))
    tokens = [summaries[m].get("avg_prompt_tokens", 0) for m in METHODS]

    bars = plt.bar(x, tokens, color="#06b6d4", width=0.5)
    plt.xlabel("Method", fontweight="bold")
    plt.ylabel("Average Prompt Tokens", fontweight="bold")
    plt.title("Prompt Token Footprint across Methods", fontweight="bold")
    plt.xticks(x, METHOD_LABELS, rotation=15)
    plt.ylim(0, max(tokens) * 1.2)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 4, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "prompt_tokens_comparison.png", dpi=300)
    plt.close()

def plot_safety(summaries):
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(METHODS))
    dang_counts = [summaries[m].get("dangerous_count", 0) for m in METHODS]

    bars = plt.bar(x, dang_counts, color="#ef4444", width=0.5)
    plt.xlabel("Method", fontweight="bold")
    plt.ylabel("Dangerous Outputs (out of 15)", fontweight="bold")
    plt.title("Safety Stress Test: Generation of Destructive Shell Commands", fontweight="bold")
    plt.xticks(x, METHOD_LABELS, rotation=15)
    plt.ylim(0, 16)
    plt.axhline(15, color="gray", linestyle=":", label="Total Risky Prompts (15)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / 15.0) * 100
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h}/15 ({pct:.0f}%)", ha="center", va="bottom", fontsize=9)

    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "safety_test_comparison.png", dpi=300)
    plt.close()

def plot_loss_curves():
    lora_path = BASE_DIR / "adapters" / "lora_0.5b" / "training_metrics.json"
    qlora_path = BASE_DIR / "adapters" / "qlora_0.5b" / "training_metrics.json"

    if not lora_path.exists() or not qlora_path.exists():
        print("Training metrics not found, skipping loss curves.")
        return

    with open(lora_path, "r", encoding="utf-8") as f:
        lora_m = json.load(f)
    with open(qlora_path, "r", encoding="utf-8") as f:
        qlora_m = json.load(f)

    epochs = [1, 2]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, lora_m["loss_history"], "o-", label="LoRA (Train Loss)", color="#2563eb", linewidth=2)
    plt.plot(epochs, lora_m["val_history"], "s--", label="LoRA (Val Loss)", color="#60a5fa", linewidth=2)
    plt.plot(epochs, qlora_m["loss_history"], "o-", label="QLoRA (Train Loss)", color="#dc2626", linewidth=2)
    plt.plot(epochs, qlora_m["val_history"], "s--", label="QLoRA (Val Loss)", color="#f87171", linewidth=2)

    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Loss (Cross-Entropy)", fontweight="bold")
    plt.title("Fine-Tuning Loss Convergence: LoRA vs QLoRA (2 Epochs)", fontweight="bold")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_loss_curves.png", dpi=300)
    plt.close()

def build_summary_table(summaries):
    rows = []
    for m in METHODS:
        s = summaries[m]
        rows.append({
            "Method": s.get("method", m),
            "Test A Exact Match (%)": round(s.get("exact_match_pct", 0), 2),
            "Test A Cmd Match (%)": round(s.get("cmd_name_match_pct", 0), 2),
            "Test A Token F1": round(s.get("mean_token_f1", 0), 4),
            "Test A Syntax (%)": round(s.get("syntax_valid_pct", 0), 2),
            "Test B Exec Acc (%)": round(s.get("exec_accuracy_pct", 0), 2),
            "Test C Dangerous (%)": round(s.get("dangerous_pct", 0), 1),
            "Latency (ms)": round(s.get("avg_latency_ms", 0), 1),
            "Prompt Tokens": round(s.get("avg_prompt_tokens", 0), 1),
            "Peak VRAM (MB)": round(s.get("peak_vram_mb", 0), 1),
            "Train Time (s)": round(s.get("train_time_sec", 0), 1),
            "Adapter Size (MB)": round(s.get("adapter_size_mb", 0), 2),
        })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    
    md_path = RESULTS_DIR / "summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"Saved summary tables to {csv_path} and {md_path}")
    print("\nSummary Table:")
    print(df.to_string(index=False))

def main():
    summaries = load_data()
    if len(summaries) < len(METHODS):
        print(f"Only {len(summaries)}/{len(METHODS)} summaries loaded yet.")
    plot_accuracy(summaries)
    plot_latency(summaries)
    plot_vram(summaries)
    plot_prompt_tokens(summaries)
    plot_safety(summaries)
    plot_loss_curves()
    build_summary_table(summaries)
    print("All figures and tables generated successfully!")

if __name__ == "__main__":
    main()


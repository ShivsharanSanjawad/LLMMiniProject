import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_web_ui(output_path="figures/web_ui_screenshot.png"):
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Header
    ax.text(5, 94, "⚡ NL2Shell Web Dashboard", fontsize=18, fontweight='bold', color='#38bdf8', va='center')
    ax.text(5, 90, "Natural Language to POSIX Bash Translation using Qwen2.5-0.5B-Instruct + LoRA", fontsize=10, color='#94a3b8', va='center')

    # Badges
    bbox_gpu = dict(boxstyle="round,pad=0.4", fc="#14532d", ec="#22c55e", lw=1)
    ax.text(95, 94, "GPU: RTX 3050 (6GB VRAM)", fontsize=9, color="#4ade80", bbox=bbox_gpu, ha='right', va='center', weight='bold')

    bbox_guard = dict(boxstyle="round,pad=0.4", fc="#1e293b", ec="#38bdf8", lw=1)
    ax.text(70, 94, "Guardrail: ACTIVE", fontsize=9, color="#38bdf8", bbox=bbox_guard, ha='right', va='center', weight='bold')

    # Divider line
    ax.plot([5, 95], [86, 86], color='#334155', lw=1.2)

    # 4 Metric Cards
    metrics = [
        ("TOTAL REQUESTS", "7", "Queries served", "#f8fafc"),
        ("AVERAGE LATENCY", "693.8 ms", "Mean inference speed", "#38bdf8"),
        ("P95 TAIL LATENCY", "1124.5 ms", "95th percentile SLA", "#f8fafc"),
        ("SECURITY BLOCKED", "1", "Lethal commands flagged", "#22c55e")
    ]
    card_w = 21
    gap = 2.3
    start_x = 5
    for i, (label, val, sub, col) in enumerate(metrics):
        cx = start_x + i * (card_w + gap)
        rect = patches.FancyBboxPatch((cx, 72), card_w, 11, boxstyle="round,pad=0.6",
                                     fc="#1e293b", ec="#334155", lw=1)
        ax.add_patch(rect)
        ax.text(cx + 2, 80.5, label, fontsize=8, color="#94a3b8", weight='bold')
        ax.text(cx + 2, 76.5, val, fontsize=14, color=col, weight='bold', family='monospace')
        ax.text(cx + 2, 73.5, sub, fontsize=7.5, color="#64748b")

    # Main Workspace Card
    ws_rect = patches.FancyBboxPatch((5, 18), 90, 50, boxstyle="round,pad=0.8",
                                     fc="#1e293b", ec="#334155", lw=1.2)
    ax.add_patch(ws_rect)

    ax.text(8, 65, "Natural Language Shell Request:", fontsize=11, color="#f8fafc", weight='bold')

    # Input Box
    inp_rect = patches.FancyBboxPatch((8, 52), 84, 10, boxstyle="round,pad=0.5",
                                      fc="#0f172a", ec="#38bdf8", lw=1.5)
    ax.add_patch(inp_rect)
    ax.text(10, 57, "how to list the directories", fontsize=11, color="#f8fafc", family='monospace')

    # Quick chips
    chips = ["List directories", "Find logs > 10MB", "Count Python lines", "Security Test (rm -rf)"]
    chip_x = 8
    for chip in chips:
        ch_box = dict(boxstyle="round,pad=0.35", fc="#334155", ec="#475569", lw=0.8)
        ax.text(chip_x, 47, chip, fontsize=8, color="#cbd5e1", bbox=ch_box, va='center')
        chip_x += len(chip) * 1.05 + 4

    # Translate Button
    btn_box = dict(boxstyle="round,pad=0.5", fc="#0284c7", ec="#38bdf8", lw=1)
    ax.text(88, 47, "Translate to Bash ->", fontsize=9.5, color="#ffffff", bbox=btn_box, ha='right', va='center', weight='bold')

    # Output Terminal
    term_rect = patches.FancyBboxPatch((8, 22), 84, 20, boxstyle="round,pad=0.6",
                                       fc="#090d16", ec="#334155", lw=1.2)
    ax.add_patch(term_rect)

    # Terminal Header bar
    ax.plot([8, 92], [38, 38], color='#1e293b', lw=1)
    # 3 dots
    ax.plot(10, 39.5, marker='o', markersize=5, color='#ef4444')
    ax.plot(11.5, 39.5, marker='o', markersize=5, color='#f59e0b')
    ax.plot(13, 39.5, marker='o', markersize=5, color='#10b981')

    # Badges in terminal header
    lat_box = dict(boxstyle="round,pad=0.2", fc="#090d16", ec="none")
    ax.text(78, 39.5, "Latency: 693.8 ms", fontsize=8, color="#94a3b8", bbox=lat_box, family='monospace', va='center')
    safe_box = dict(boxstyle="round,pad=0.3", fc="#14532d", ec="#22c55e", lw=0.8)
    ax.text(90, 39.5, "PASSED (SAFE)", fontsize=7.5, color="#4ade80", bbox=safe_box, ha='right', va='center', weight='bold')

    # Command text
    ax.text(11, 29, "$", fontsize=15, color="#64748b", weight='bold', family='monospace', va='center')
    ax.text(13.5, 29, "find . -type d", fontsize=15, color="#38bdf8", weight='bold', family='monospace', va='center')

    copy_box = dict(boxstyle="round,pad=0.35", fc="#1e293b", ec="#475569", lw=0.8)
    ax.text(90, 29, "Copy", fontsize=8.5, color="#cbd5e1", bbox=copy_box, ha='right', va='center')

    # Footer
    ax.text(50, 8, "FastAPI Service • Served at http://localhost:8000 • Live AST Guardrail Validation",
            fontsize=8.5, color="#64748b", ha='center')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    draw_web_ui("figures/web_ui_dashboard.png")

"""
Read results/part3/results.json (written by run_part3.py) and produce:
  * comparison table  -> comparison.csv and comparison.md
  * plots             -> ppl_vs_acc.png, beam_time.png, metrics_by_config.png

    python compare_part3.py --results results/part3/results.json --out_dir results/part3
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"greedy": "tab:blue", "beam": "tab:orange", "top_k": "tab:green", "top_p": "tab:red"}


def family(label):
    for f in ("greedy", "beam", "top_k", "top_p"):
        if label.startswith(f):
            return f
    return "reference"


def pretty(label):
    """beam_width4 -> beam (w=4), top_k_k20 -> top-k (k=20), top_p_p0.9 -> top-p (p=0.9)"""
    if label.startswith("beam_width"):
        return f"beam (w={label[len('beam_width'):]})"
    if label.startswith("top_k_k"):
        return f"top-k (k={label[len('top_k_k'):]})"
    if label.startswith("top_p_p"):
        return f"top-p (p={label[len('top_p_p'):]})"
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/part3/results.json")
    ap.add_argument("--out_dir", default="results/part3")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)["results"]
    os.makedirs(args.out_dir, exist_ok=True)

    cols = ["ppl", "token_acc", "distinct_2", "total_sec", "sec_per_sample"]
    labels = list(results.keys())

    # ---------------- table ----------------
    def fmt(v):
        return "" if v is None else f"{v:.4f}"

    with open(f"{args.out_dir}/comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config"] + cols)
        for l in labels:
            w.writerow([pretty(l)] + [fmt(results[l].get(c)) for c in cols])

    with open(f"{args.out_dir}/comparison.md", "w") as f:
        f.write("| config | " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * (len(cols) + 1) + "\n")
        for l in labels:
            f.write(f"| {pretty(l)} | " + " | ".join(fmt(results[l].get(c)) for c in cols) + " |\n")

    decoded = [l for l in labels if family(l) != "reference"]

    # ---------------- plot 1: perplexity vs accuracy ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for l in decoded:
        r = results[l]
        ax.scatter(r["ppl"], r["token_acc"], color=COLORS[family(l)], s=60)
        ax.annotate(pretty(l), (r["ppl"], r["token_acc"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ref = results.get("reference (teacher-forced)")
    if ref:
        ax.scatter(ref["ppl"], ref["token_acc"], marker="*", color="black", s=120, label="reference (teacher-forced)")
    for fam, c in COLORS.items():
        ax.scatter([], [], color=c, label=fam.replace("_", "-"))
    ax.set_xlabel("perplexity of generated tokens (log scale)")
    ax.set_ylabel("token accuracy vs reference")
    ax.set_xscale("log")
    ax.set_title("Perplexity vs accuracy by decoding strategy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/ppl_vs_acc.png", dpi=200)
    plt.close(fig)

    # ---------------- plot 2: beam search time vs width ----------------
    beams = sorted((l for l in labels if family(l) == "beam"), key=lambda l: int(l[len("beam_width"):]))
    if beams:
        widths = [int(l[len("beam_width"):]) for l in beams]
        secs = [results[l]["sec_per_sample"] for l in beams]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].bar([str(w) for w in widths], secs, color=COLORS["beam"])
        axes[0].set_xlabel("beam width")
        axes[0].set_ylabel("seconds per sample")
        axes[0].set_title("Beam search time per sample")
        axes[1].plot(widths, secs, marker="o", color=COLORS["beam"], label="measured")
        axes[1].plot(widths, [secs[0] * w / widths[0] for w in widths], "--", color="gray", label="linear in width (from w=1)")
        axes[1].set_xlabel("beam width")
        axes[1].set_ylabel("seconds per sample")
        axes[1].set_title("Scaling with beam width")
        axes[1].legend(fontsize=8)
        for a in axes:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/beam_time.png", dpi=200)
        plt.close(fig)

    # ---------------- plot 3: all metrics by config ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, title) in zip(axes, [("ppl", "Perplexity"), ("token_acc", "Token accuracy"), ("distinct_2", "Distinct-2")]):
        vals = [results[l][metric] for l in decoded]
        ax.bar(range(len(decoded)), vals, color=[COLORS[family(l)] for l in decoded])
        ax.set_xticks(range(len(decoded)))
        ax.set_xticklabels([pretty(l) for l in decoded], rotation=60, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if ref and metric in ref:
            ax.axhline(ref[metric], color="black", linestyle="--", linewidth=1, label="reference")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/metrics_by_config.png", dpi=200)
    plt.close(fig)

    print(open(f"{args.out_dir}/comparison.md").read())
    print(f"Saved tables and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def method_from_filename(path: str) -> str:
    name = os.path.basename(path).lower()
    if "argmax_wt" in name:
        return "argmax_wt"
    if "argmax" in name:
        return "argmax"
    return "unknown"


def load_eval_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def action_fractions(action_distribution: List[Dict], num_actions: int = 5) -> Dict[int, float]:
    fractions = {action_id: 0.0 for action_id in range(num_actions)}
    for entry in action_distribution:
        action_id = int(entry.get("action_id", -1))
        frac = float(entry.get("frac", 0.0))
        if action_id in fractions:
            fractions[action_id] = frac
    return fractions


def write_summary_csv(rows: List[Dict], out_path: str) -> None:
    all_keys = set().union(*(row.keys() for row in rows))
    preferred_order = [
        "timestamp_utc",
        "method",
        "slo_profile",
        "baseline1_avg_reward",
        "bestfixed_action_id",
        "bestfixed_avg_reward",
        "bestfixed_avg_accuracy",
        "bestfixed_avg_cost_tokens",
        "learned_avg_reward",
        "learned_avg_accuracy",
        "learned_avg_cost_tokens",
        "learned_hallucination_rate",
        "learned_refusal_rate",
        "learned_refusal_overrides",
        "label",
    ]
    fieldnames = [key for key in preferred_order if key in all_keys]
    remaining = sorted(all_keys - set(fieldnames))
    fieldnames.extend(remaining)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_action_distribution(rows: List[Dict], out_path: str) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))

    for action_id in range(5):
        offsets = [xi + (action_id - 2) * width for xi in x]
        values = [row[f"frac_action_{action_id}"] for row in rows]
        ax.bar(offsets, values, width=width, label=f"action_{action_id}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Action Distribution by SLO Profile and Method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cost_vs_accuracy(rows: List[Dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = {"learned": False, "best_fixed": False}

    for row in rows:
        label = row["label"]
        x_learned = row["learned_avg_cost_tokens"]
        y_learned = row["learned_avg_accuracy"]
        ax.scatter(
            x_learned,
            y_learned,
            marker="o",
            label="learned_policy" if not plotted["learned"] else None,
        )
        plotted["learned"] = True
        ax.text(x_learned, y_learned, label, fontsize=8)

        x_best = row["bestfixed_avg_cost_tokens"]
        y_best = row["bestfixed_avg_accuracy"]
        ax.scatter(
            x_best,
            y_best,
            marker="s",
            label="best_fixed_action_baseline" if not plotted["best_fixed"] else None,
        )
        plotted["best_fixed"] = True
        ax.text(x_best, y_best, label, fontsize=8)

    ax.set_xlabel("avg_cost_tokens")
    ax.set_ylabel("avg_accuracy")
    ax.set_title("Cost vs Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reward_vs_baseline(rows: List[Dict], out_path: str) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.35

    learned = [row["learned_avg_reward"] for row in rows]
    best_fixed = [row["bestfixed_avg_reward"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([xi - width / 2 for xi in x], best_fixed, width=width, label="best_fixed")
    ax.bar([xi + width / 2 for xi in x], learned, width=width, label="learned")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("avg_reward")
    ax.set_title("Reward vs Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary CSV and figures from eval JSON outputs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files or glob patterns.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for CSV and figures.",
    )
    args = parser.parse_args()

    input_paths = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        if matches:
            input_paths.extend(matches)
        else:
            input_paths.append(pattern)

    input_paths = [p for p in input_paths if os.path.isfile(p)]
    if not input_paths:
        raise FileNotFoundError("No input JSON files found.")

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for path in input_paths:
        payload = load_eval_json(path)
        meta = payload.get("meta", {})
        method = method_from_filename(path)
        slo_profile = meta.get("slo_profile", "unknown")
        label = f"{slo_profile}\n{method}"

        baseline = payload.get("baseline_fixed_action_id_1", {})
        best_fixed = payload.get("best_fixed_action_baseline", {})
        learned = payload.get("learned_policy", {})

        baseline_metrics = baseline.get("metrics", {})
        best_metrics = best_fixed.get("metrics", {})
        learned_metrics = learned.get("metrics", {})
        action_dist = learned.get("action_distribution", [])
        fractions = action_fractions(action_dist, num_actions=5)

        row = {
            "timestamp_utc": meta.get("timestamp_utc", ""),
            "method": method,
            "slo_profile": slo_profile,
            "baseline1_avg_reward": float(baseline_metrics.get("avg_reward", 0.0)),
            "baseline1_retrieval_hit_rate": float(
                baseline_metrics.get("retrieval_hit_rate", 0.0)
            ),
            "bestfixed_action_id": best_fixed.get("best_action_id", None),
            "bestfixed_avg_reward": float(best_metrics.get("avg_reward", 0.0)),
            "bestfixed_retrieval_hit_rate": float(
                best_metrics.get("retrieval_hit_rate", 0.0)
            ),
            "learned_avg_reward": float(learned_metrics.get("avg_reward", 0.0)),
            "learned_avg_accuracy": float(learned_metrics.get("avg_accuracy", 0.0)),
            "learned_avg_cost_tokens": float(learned_metrics.get("avg_cost_tokens", 0.0)),
            "learned_hallucination_rate": float(
                learned_metrics.get("hallucination_rate", 0.0)
            ),
            "learned_refusal_rate": float(learned_metrics.get("refusal_rate", 0.0)),
            "learned_refusal_overrides": int(learned.get("refusal_overrides", 0)),
            "learned_retrieval_hit_rate": float(
                learned_metrics.get("retrieval_hit_rate", 0.0)
            ),
            "frac_action_0": float(fractions[0]),
            "frac_action_1": float(fractions[1]),
            "frac_action_2": float(fractions[2]),
            "frac_action_3": float(fractions[3]),
            "frac_action_4": float(fractions[4]),
            "bestfixed_avg_cost_tokens": float(best_metrics.get("avg_cost_tokens", 0.0)),
            "bestfixed_avg_accuracy": float(best_metrics.get("avg_accuracy", 0.0)),
            "label": label,
        }
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "summary.csv")
    write_summary_csv(rows, csv_path)

    action_path = os.path.join(args.out_dir, "action_distribution.png")
    cost_path = os.path.join(args.out_dir, "cost_vs_accuracy.png")
    reward_path = os.path.join(args.out_dir, "reward_vs_baseline.png")

    plot_action_distribution(rows, action_path)
    plot_cost_vs_accuracy(rows, cost_path)
    plot_reward_vs_baseline(rows, reward_path)

    print(f"Wrote summary CSV to {csv_path}")
    print(f"Wrote figure to {action_path}")
    print(f"Wrote figure to {cost_path}")
    print(f"Wrote figure to {reward_path}")


if __name__ == "__main__":
    main()

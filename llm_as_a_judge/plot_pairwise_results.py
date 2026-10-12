import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from criteria_pairwise import CRITERIA_PAIRWISE

UNPARSED = "Unparsed"


def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def build_summary(df: pd.DataFrame) -> pd.DataFrame: #One row per (dataset, model_a, model_b, criterion, verdict) with count and percentage. 
    summary_rows = []
    for (dataset, model_a, model_b), group in df.groupby(["dataset", "model_a", "model_b"]):
        total = len(group)
        verdict_labels = {"response_a": model_a, "response_b": model_b, "tie": "Tie", "neither": "Neither"} #first, we tried to use neither; however the models couldn't distinguish between neither and tie, so we changed the prompt to not use neither in the eval
        categories = list(verdict_labels.values()) + [UNPARSED] #categories + the unparsed case if nothing valid has been extracted

        for key, label, _ in CRITERIA_PAIRWISE: #For each question, if the verdict is valid, translate "response_a"/"response_b"/"tie"/"neither"; otherwise, mark it as "Unparsed"
            verdicts = group.apply(
                lambda row: verdict_labels.get(row[key], UNPARSED) if is_valid_flag(row[f"{key}_valid"]) else UNPARSED,
                axis=1,
            )
            counts = verdicts.value_counts()
            for option in categories:
                count = int(counts.get(option, 0))
                summary_rows.append({
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "criterion": label,
                    "verdict": option,
                    "count": count,
                    "percentage": round(100 * count / total, 1) if total else 0.0,
                })
    return pd.DataFrame(summary_rows)


def build_criterion_table(criterion_summary: pd.DataFrame, categories: list) -> pd.DataFrame: #Table for a single (dataset, pair, criterion): one row, columns=verdict categories
    row = {}
    for cat in categories:
        match = criterion_summary[criterion_summary["verdict"] == cat]
        count = int(match["count"].sum())
        pct = float(match["percentage"].sum()) #calculate the percentage for each category
        row[cat] = f"{count} ({pct:.1f}%)"
    return pd.DataFrame([row])[categories]


def print_and_save_tables(summary: pd.DataFrame, output_dir: str):
    for (dataset, model_a, model_b), pair_summary in summary.groupby(["dataset", "model_a", "model_b"]):
        pair_dir_name = f"{model_a}_vs_{model_b}"
        tables_dir = os.path.join(output_dir, dataset, pair_dir_name, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        print(f"\n {dataset}: {model_a} vs {model_b}")
        for key, label, _ in CRITERIA_PAIRWISE:
            criterion_summary = pair_summary[pair_summary["criterion"] == label]
            # Only categories that actually occurred at least once 
            nonzero = set(criterion_summary.loc[criterion_summary["count"] > 0, "verdict"])
            ordered = [c for c in (model_a, model_b, "Tie", "Neither", UNPARSED) if c in nonzero]
            if not ordered:
                ordered = [model_a, model_b, "Tie", "Neither", UNPARSED]
            table = build_criterion_table(criterion_summary, ordered)

            print(f"\n--- {label} ---")
            print(table.to_string(index=False))
            table.to_json(os.path.join(tables_dir, f"{key}.json"), orient="records", indent=2)


def plot_criteria(summary: pd.DataFrame, output_dir: str): #One grouped bar chart per criterion: bars = percentage of votes (%)
    for (dataset, model_a, model_b), pair_summary in summary.groupby(["dataset", "model_a", "model_b"]):
        pair_dir_name = f"{model_a}_vs_{model_b}"
        plots_dir = os.path.join(output_dir, dataset, pair_dir_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for key, label, _ in CRITERIA_PAIRWISE:
            criterion_summary = pair_summary[pair_summary["criterion"] == label]
            # Only categories that actually occurred at least once 
            nonzero = set(criterion_summary.loc[criterion_summary["count"] > 0, "verdict"])
            categories = [c for c in (model_a, model_b, "Tie", "Neither", UNPARSED) if c in nonzero]
            if not categories:
                categories = [model_a, model_b, "Tie", "Neither", UNPARSED]
            values = [criterion_summary.loc[criterion_summary["verdict"] == cat, "percentage"].sum() for cat in categories]

            fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.8), 5))
            bars = ax.bar(range(len(categories)), values, width=0.6)
            ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=20, ha="right")
            ax.set_ylabel("% of questions")
            ax.set_ylim(0, 110)
            ax.set_title(f"{label} ({dataset}: {model_a} vs {model_b})", pad=12)
            fig.tight_layout()

            fig.savefig(os.path.join(plots_dir, f"{key}.png"), dpi=150)
            plt.close(fig)


_STACKED_COLORS = {"a": "#2a78d6", "tie": "#c3c2b7", "neither": "#898781", "b": "#e34948"}
_CHART_SURFACE = "#fcfcfb"

_DISPLAY_NAME_OVERRIDES = {
    "Mixed_lr1.1247e-4_rank16": "r16_lr1.1247e-4",
    "Vanilla_P2_lr3.5566e-5_rank32": "P2_r32_3.5566e-5",
}

def _display_name(name: str) -> str:
    return _DISPLAY_NAME_OVERRIDES.get(name, name)


def plot_stacked_summary(summary: pd.DataFrame, output_dir: str): #One combined horizontal bar chart per (dataset, pair) with all 7 criteria 
    for (dataset, model_a, model_b), pair_summary in summary.groupby(["dataset", "model_a", "model_b"]):
        pair_dir_name = f"{model_a}_vs_{model_b}"
        out_dir = os.path.join(output_dir, dataset, pair_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        labels = [label for _, label, _ in CRITERIA_PAIRWISE]
        categories_all = [model_a, "Tie", "Neither", model_b]
        display_labels_all = [_display_name(model_a), "Tie", "Neither", _display_name(model_b)]
        colors_all = [_STACKED_COLORS["a"], _STACKED_COLORS["tie"], _STACKED_COLORS["neither"], _STACKED_COLORS["b"]]

       # Only categories that actually occurred at least once 
        nonzero_cats = set(pair_summary.loc[pair_summary["count"] > 0, "verdict"])
        keep = [i for i, cat in enumerate(categories_all) if cat in nonzero_cats]
        if not keep:
            keep = list(range(len(categories_all)))
        categories = [categories_all[i] for i in keep]
        display_labels = [display_labels_all[i] for i in keep]
        colors = [colors_all[i] for i in keep]

        fig, ax = plt.subplots(figsize=(9, 1 + 0.7 * len(labels)))
        y = list(range(len(labels)))
        left = [0.0] * len(labels)
        for cat, disp_label, color in zip(categories, display_labels, colors):
            values = [
                pair_summary.loc[(pair_summary["criterion"] == label) & (pair_summary["verdict"] == cat), "percentage"].sum()
                for label in labels
            ]
            ax.barh(y, values, left=left, height=0.6, label=disp_label, color=color,
                    edgecolor=_CHART_SURFACE, linewidth=1.5)
            left = [l + v for l, v in zip(left, values)]

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  
        ax.set_xlim(0, 100)
        ax.set_xlabel("Distribution of Votes (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False, fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(left=False)

        fig.suptitle(f"{_display_name(model_a)} vs. {_display_name(model_b)} ({dataset})", fontsize=13, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.90)) 

        fig.savefig(os.path.join(out_dir, "combined_summary.png"), dpi=150, facecolor=_CHART_SURFACE)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate pairwise llm-as-a-judge results into win-rate percentages and charts")
    parser.add_argument("--results_paths", type=str, nargs="+", required=True,
                         help="One or more *_judge_pairwise_evaluation.json files from extract_pairwise_scores.py.")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    all_rows = []
    for path in args.results_paths:
        all_rows.extend(load_results(path))
    df = pd.DataFrame(all_rows)

    if df["dataset"].nunique() > 1: #combined eval
        combined = df.copy() 
        combined["dataset"] = "Combined"
        df = pd.concat([df, combined], ignore_index=True) 

    summary = build_summary(df)
    os.makedirs(args.output_dir, exist_ok=True)

    for (dataset, model_a, model_b), pair_summary in summary.groupby(["dataset", "model_a", "model_b"]):
        pair_dir_name = f"{model_a}_vs_{model_b}"
        out_dir = os.path.join(args.output_dir, dataset, pair_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        pair_summary.to_json(os.path.join(out_dir, "summary.json"), orient="records", indent=2)

    print_and_save_tables(summary, args.output_dir)
    plot_criteria(summary, args.output_dir)
    plot_stacked_summary(summary, args.output_dir)
    print(f"\nSaved summaries and plots under {args.output_dir}")


if __name__ == "__main__":
    main()

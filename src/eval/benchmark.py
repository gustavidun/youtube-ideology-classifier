"""
Benchmark evaluation of LLM teacher models against ground-truth human judgments.
Generates plots/leaderboard.png and plots/leaderboard.csv.
"""

from pathlib import Path
from elo_rating import Elo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import accuracy_score

from src.config import DATA_DIR, ROOT

EVAL_DIR = DATA_DIR / "out" / "eval" / "teacher"
HUMAN_LABELS_PATH = DATA_DIR / "human_labels" / "human_labels.pkl"
PLOTS_DIR = ROOT / "plots"
OUTPUT_PLOT = PLOTS_DIR / "leaderboard.png"
OUTPUT_DATA = PLOTS_DIR / "leaderboard.parquet"


def evaluate_scores(score_dict: dict, human_labels: pd.DataFrame):
    df_eval = human_labels.copy()
    df_eval["top"] = df_eval["top_video_id"].map(score_dict)
    df_eval["bot"] = df_eval["bottom_video_id"].map(score_dict)
    df_eval = df_eval.dropna(subset=["top", "bot"])
    if len(df_eval) == 0:
        return 0.0, 0

    pred_is_top = (df_eval["top"] > df_eval["bot"]) == (
        df_eval["category_name"] == "Indicate which video is more conservative"
    )
    pred_label = np.where(pred_is_top, "top", "bot")
    acc = round(accuracy_score(df_eval["human_label"], pred_label) * 100, 2)
    return acc, len(df_eval)


def run_benchmark():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    human_labels = pd.read_pickle(HUMAN_LABELS_PATH)

    records = []
    for m_dir in sorted(EVAL_DIR.iterdir()):
        if not m_dir.is_dir():
            continue
        model_name = m_dir.name

        # 1. Ranking (Elo)
        for ctx in ["context", "no_context"]:
            p = m_dir / f"{ctx}_ranking.parquet"
            if not p.exists():
                continue
            df_rank = pl.read_parquet(p).to_pandas()
            for dim in ["overall", "economic", "cultural"]:
                col = f"is_more_right_leaning_{dim}"
                if col not in df_rank.columns:
                    continue
                elo = Elo(denom=1.0)
                for _, row in df_rank.iterrows():
                    elo.add_match(
                        str(row["id_a"]),
                        str(row["id_b"]),
                        result=float(row[col]),
                        k=0.1,
                    )
                acc, pairs = evaluate_scores(elo.ratings(), human_labels)
                records.append({
                    "Model": model_name,
                    "Method": "Ranking (Elo)",
                    "Dimension": dim.title(),
                    "Context": "With Context" if ctx == "context" else "No Context",
                    "Setup": f"{model_name}  |  Ranking ({dim.title()})",
                    "Accuracy (%)": acc,
                    "Pairs": pairs,
                })

        # 2. Absolute (Direct)
        for ctx in ["context", "no_context"]:
            p = m_dir / f"{ctx}_absolute.parquet"
            if not p.exists():
                continue
            df_abs = pl.read_parquet(p).to_pandas()
            dims = [
                ("Overall", "overall_score"),
                ("Economic", "economic_score"),
                ("Cultural", "galtan_score"),
            ]
            for dim, col in dims:
                if col not in df_abs.columns:
                    continue
                score_dict = df_abs.groupby("id_a")[col].mean().to_dict()
                acc, pairs = evaluate_scores(score_dict, human_labels)
                records.append({
                    "Model": model_name,
                    "Method": "Absolute (Direct)",
                    "Dimension": dim,
                    "Context": "With Context" if ctx == "context" else "No Context",
                    "Setup": f"{model_name}  |  Absolute ({dim})",
                    "Accuracy (%)": acc,
                    "Pairs": pairs,
                })

    if not records:
        return

    df = pd.DataFrame(records)

    # Plot with pandas + seaborn
    order = (
        df.groupby("Setup")["Accuracy (%)"]
        .max()
        .sort_values(ascending=False)
        .index
    )

    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid", font_scale=1.0)
    ax = sns.barplot(
        data=df,
        y="Setup",
        x="Accuracy (%)",
        hue="Context",
        order=order,
        palette={"With Context": "#2b5c8f", "No Context": "#d95f02"},
    )
    ax.set_xlim(58, 75)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(
        "Teacher Ideology Benchmark: Accuracy by Condition (Ranked)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Accuracy (%) vs. Human Labels", fontweight="bold")
    ax.set_ylabel("")

    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=3, fontsize=9)

    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

    df.to_csv(OUTPUT_DATA.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    run_benchmark()

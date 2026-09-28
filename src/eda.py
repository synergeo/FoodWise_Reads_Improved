"""Create reproducible EDA and model-evaluation figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_figures() -> None:
    sns.set_theme(style="whitegrid", palette="crest")
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    books = pd.read_csv("data/clustered_books.csv")
    metrics = json.loads(Path("models/metrics.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    books["source"].value_counts().plot(kind="bar", ax=ax, color=["#d97706", "#0f766e"])
    ax.set(title="Books retained by data source", xlabel="", ylabel="Books")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output / "source_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    books["known_domain"].value_counts().sort_values().plot(kind="barh", ax=ax, color="#0f766e")
    ax.set(title="Balanced modelling domains", xlabel="Books", ylabel="")
    fig.tight_layout()
    fig.savefig(output / "topic_counts.png", dpi=180)
    plt.close(fig)

    selection = pd.read_csv(output / "model_selection.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(selection["components"], selection["silhouette"], marker="o", color="#0f766e", linewidth=2)
    ax.axhline(0.50, color="#64748b", linestyle=":", label="Target = 0.50")
    ax.axvline(metrics["selected_svd_components"], color="#d97706", linestyle="--", label=f"Selected SVD={metrics['selected_svd_components']}")
    ax.set(title="Representation selection at k=6", xlabel="SVD components", ylabel="Silhouette score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "silhouette_scores.png", dpi=180)
    plt.close(fig)

    pca = pd.read_csv(output / "pca_coordinates.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pca, x="PC1", y="PC2", hue="known_domain", palette="Set2", s=34, alpha=.8, ax=ax)
    ax.set_title("PCA projection of the improved food-book model")
    ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "pca_clusters.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    create_figures()

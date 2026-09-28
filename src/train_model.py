"""Train the improved, leakage-controlled food-book clustering model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


TEXT_COLUMN = "book_text"
N_CLUSTERS = 6
BOOKS_PER_DOMAIN = 100
RANDOM_STATE = 42
STABILITY_SEEDS = (11, 22, 33, 42, 55)

DOMAIN_MAP = {
    "baking and desserts": "Baking and Pastry",
    "baking": "Baking and Pastry",
    "nutrition": "Nutrition",
    "nutrition and healthy eating": "Nutrition",
    "food safety": "Food Safety",
    "agriculture": "Agriculture",
    "sustainable agriculture": "Agriculture",
    "fermentation": "Fermentation",
    "food business": "Food Business",
}


def _remove_collection_label(subjects: object, source_topic: str) -> str:
    text = "" if pd.isna(subjects) else str(subjects)
    text = re.sub(re.escape(source_topic), " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" |")


def clean_books(raw: pd.DataFrame, balance: bool = True) -> pd.DataFrame:
    """Clean, consolidate six domains and optionally create a balanced catalogue."""
    required = {"title", "authors", "subjects", "source", "source_topic"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    books = raw.copy()
    for column in ["title", "authors", "subjects", "languages", "source_topic"]:
        if column not in books:
            books[column] = ""
        books[column] = books[column].fillna("").astype(str).str.strip()
    books = books[(books["title"] != "") & (books["authors"] != "")]
    books = books[books["source_topic"].isin(DOMAIN_MAP)].copy()
    books["known_domain"] = books["source_topic"].map(DOMAIN_MAP)
    books["subjects_clean"] = books.apply(
        lambda row: _remove_collection_label(row["subjects"], row["source_topic"]), axis=1
    )

    for column in ["average_rating", "rating_count", "first_publish_year", "edition_count"]:
        if column not in books:
            books[column] = np.nan
        books[column] = pd.to_numeric(books[column], errors="coerce")

    # Author and known-domain fields are excluded. This encourages topical
    # rather than author-identity or collection-label clusters.
    books[TEXT_COLUMN] = (
        books["title"] + " " + books["subjects_clean"]
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    duplicate_key = (
        books["title"].str.lower().str.replace(r"\W+", "", regex=True)
        + books["authors"].str.lower().str.replace(r"\W+", "", regex=True)
    )
    books = books.loc[~duplicate_key.duplicated()].copy()

    if balance:
        counts = books["known_domain"].value_counts()
        if (counts < BOOKS_PER_DOMAIN).any():
            shortage = counts[counts < BOOKS_PER_DOMAIN].to_dict()
            raise ValueError(f"Insufficient books for balanced domains: {shortage}")
        books = pd.concat(
            [
                group.sample(BOOKS_PER_DOMAIN, random_state=RANDOM_STATE)
                for _, group in books.groupby("known_domain", sort=True)
            ],
            ignore_index=True,
        )
    return books.reset_index(drop=True)


def _vectorize(books: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=10000,
        sublinear_tf=True,
    )
    return vectorizer, vectorizer.fit_transform(books[TEXT_COLUMN])


def train(catalog_path: str = "data/books_combined.csv") -> dict[str, object]:
    raw = pd.read_csv(catalog_path)
    books = clean_books(raw, balance=True)
    vectorizer, tfidf = _vectorize(books)

    selection_rows: list[dict[str, float | int]] = []
    representations: dict[int, tuple[object, np.ndarray]] = {}
    for components in (5, 6, 8, 10, 12, 15, 20):
        reducer = make_pipeline(
            TruncatedSVD(n_components=components, random_state=RANDOM_STATE),
            Normalizer(copy=False),
        )
        reduced = reducer.fit_transform(tfidf)
        labels = KMeans(
            n_clusters=N_CLUSTERS, n_init=100, random_state=RANDOM_STATE
        ).fit_predict(reduced)
        sizes = pd.Series(labels).value_counts()
        selection_rows.append({
            "components": components,
            "silhouette": float(silhouette_score(reduced, labels)),
            "ari": float(adjusted_rand_score(books["known_domain"], labels)),
            "nmi": float(normalized_mutual_info_score(books["known_domain"], labels)),
            "minimum_cluster": int(sizes.min()),
            "maximum_cluster": int(sizes.max()),
        })
        representations[components] = (reducer, reduced)

    selection = pd.DataFrame(selection_rows)
    eligible = selection[selection["minimum_cluster"] >= 30]
    selected = (eligible if not eligible.empty else selection).sort_values(
        ["silhouette", "ari"], ascending=False
    ).iloc[0]
    components = int(selected["components"])
    reducer, matrix = representations[components]

    stability: list[dict[str, float | int]] = []
    for seed in STABILITY_SEEDS:
        labels = KMeans(n_clusters=N_CLUSTERS, n_init=100, random_state=seed).fit_predict(matrix)
        stability.append({
            "seed": seed,
            "silhouette": float(silhouette_score(matrix, labels)),
            "ari": float(adjusted_rand_score(books["known_domain"], labels)),
            "nmi": float(normalized_mutual_info_score(books["known_domain"], labels)),
        })

    model = KMeans(n_clusters=N_CLUSTERS, n_init=100, random_state=RANDOM_STATE)
    books["cluster"] = model.fit_predict(matrix)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(matrix)
    pca_frame = pd.DataFrame({
        "title": books["title"],
        "known_domain": books["known_domain"],
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
        "cluster": books["cluster"],
    })

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    joblib.dump(reducer, "models/svd_reducer.joblib")
    joblib.dump(model, "models/kmeans.joblib")
    joblib.dump(pca, "models/pca.joblib")
    sparse.save_npz("models/book_matrix.npz", sparse.csr_matrix(matrix))
    books.to_csv("data/clustered_books.csv", index=False, encoding="utf-8-sig")
    pca_frame.to_csv("outputs/pca_coordinates.csv", index=False, encoding="utf-8-sig")
    selection.to_csv("outputs/model_selection.csv", index=False)
    pd.DataFrame(stability).to_csv("outputs/stability_scores.csv", index=False)
    pd.crosstab(books["cluster"], books["known_domain"]).to_csv(
        "outputs/cluster_domain_table.csv", encoding="utf-8-sig"
    )

    stable = pd.DataFrame(stability)
    metrics = {
        "books_collected": int(len(raw)),
        "books_modelled": int(len(books)),
        "domains": int(books["known_domain"].nunique()),
        "selected_k": N_CLUSTERS,
        "selected_svd_components": components,
        "silhouette": round(float(selected["silhouette"]), 4),
        "ari_vs_heldout_domain": round(float(selected["ari"]), 4),
        "nmi_vs_heldout_domain": round(float(selected["nmi"]), 4),
        "stability_silhouette_min": round(float(stable["silhouette"].min()), 4),
        "stability_silhouette_mean": round(float(stable["silhouette"].mean()), 4),
        "cluster_sizes": {
            str(key): int(value)
            for key, value in books["cluster"].value_counts().sort_index().items()
        },
        "domain_sizes": {
            key: int(value)
            for key, value in books["known_domain"].value_counts().sort_index().items()
        },
        "pca_explained_variance": [round(float(x), 4) for x in pca.explained_variance_ratio_],
        "leakage_control": "known_domain and source_topic excluded; source-topic phrase removed from subjects",
    }
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", nargs="?", default="data/books_combined.csv")
    args = parser.parse_args()
    print(json.dumps(train(args.catalog), indent=2))


if __name__ == "__main__":
    main()

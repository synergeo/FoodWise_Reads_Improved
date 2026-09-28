"""Content-based recommendations constrained to the selected book's cluster."""

from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_book(title: str, books: pd.DataFrame, author: str | None = None) -> tuple[int | None, list[str]]:
    title_key = normalize(title)
    title_keys = books["title"].map(normalize)
    matches = books.index[title_keys == title_key].tolist()
    if author and matches:
        author_key = normalize(author)
        matches = [i for i in matches if author_key in normalize(books.loc[i, "authors"])]
    if matches:
        return matches[0], []
    close = difflib.get_close_matches(title_key, title_keys.tolist(), n=5, cutoff=0.55)
    suggestions = [books.loc[title_keys[title_keys == key].index[0], "title"] for key in close]
    return None, suggestions


def recommend_books(title: str, books: pd.DataFrame, matrix: sparse.csr_matrix, author: str | None = None, number: int = 5, topic_filter: str | None = None) -> dict[str, object]:
    index, suggestions = find_book(title, books, author)
    if index is None:
        return {"found": False, "suggestions": suggestions, "recommendations": []}
    cluster = int(books.loc[index, "cluster"])
    eligible = (books["cluster"] == cluster) & (books.index != index)
    if topic_filter and topic_filter != "All food domains":
        eligible &= books["known_domain"].fillna("").eq(topic_filter)
    candidates = books.index[eligible].to_numpy()
    if not len(candidates):
        return {"found": True, "cluster": cluster, "recommendations": []}
    similarities = cosine_similarity(matrix[index], matrix[candidates]).ravel()
    order = np.argsort(similarities)[::-1][:number]
    results = []
    for position in order:
        row = books.loc[int(candidates[position])]
        results.append({
            "title": row["title"], "authors": row["authors"],
            "subjects": row.get("subjects", ""),
            "known_domain": row.get("known_domain", ""),
            "source_topic": row.get("source_topic", ""),
            "average_rating": None if pd.isna(row.get("average_rating")) else round(float(row["average_rating"]), 2),
            "year": None if pd.isna(row.get("first_publish_year")) else int(row["first_publish_year"]),
            "similarity": round(float(similarities[position]), 3),
            "cover_url": None if pd.isna(row.get("cover_url")) else row.get("cover_url"),
            "book_url": None if pd.isna(row.get("book_url")) else row.get("book_url"),
        })
    return {
        "found": True,
        "selected": {"title": books.loc[index, "title"], "authors": books.loc[index, "authors"]},
        "cluster": cluster,
        "recommendations": results,
    }

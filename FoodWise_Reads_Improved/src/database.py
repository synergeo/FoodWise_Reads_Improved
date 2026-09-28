"""Load the cleaned catalogue into SQLite and run useful project queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def build_database(csv_path: str = "data/clustered_books.csv", database_path: str = "data/food_books.db") -> None:
    books = pd.read_csv(csv_path)
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        books.to_sql("books", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_books_cluster ON books(cluster)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_books_title ON books(title)")


def example_queries(database_path: str = "data/food_books.db") -> dict[str, pd.DataFrame]:
    with sqlite3.connect(database_path) as connection:
        return {
            "cluster_sizes": pd.read_sql_query("SELECT cluster, COUNT(*) AS books FROM books GROUP BY cluster ORDER BY books DESC", connection),
            "top_rated": pd.read_sql_query("SELECT title, authors, average_rating, rating_count FROM books WHERE rating_count >= 100 ORDER BY average_rating DESC, rating_count DESC LIMIT 10", connection),
            "sources": pd.read_sql_query("SELECT source, COUNT(*) AS books FROM books GROUP BY source", connection),
        }


if __name__ == "__main__":
    build_database()
    for name, frame in example_queries().items():
        print(f"\n{name}\n", frame.to_string(index=False))


"""Collect 500 scraped and 700 API food-book records for a 1,050+ catalogue."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


GOODREADS_URL = "https://www.goodreads.com/list/show/185.Best_Cookbooks?page={page}"
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
FOOD_TOPICS = [
    "baking", "nutrition", "food safety", "agriculture", "fermentation", "food business",
]
OPEN_LIBRARY_FIELDS = (
    "key,title,author_name,first_publish_year,subject,language,cover_i,edition_count,ratings_average,ratings_count"
)
HEADERS = {"User-Agent": "FoodBookRecommender/1.0 educational Ironhack project"}


def infer_food_topic(title: str) -> str:
    """Create a transparent broad topic from cookbook-title keywords."""
    value = title.lower()
    rules = [
        ("baking and desserts", ["bake", "baking", "bread", "cake", "dessert", "pastry", "cookie", "chocolate", "pie"]),
        ("vegetarian and plant based", ["vegan", "vegetarian", "plant-based", "plant based", "vegetable"]),
        ("nutrition and healthy eating", ["healthy", "nutrition", "diet", "keto", "paleo", "whole30", "gluten-free", "gluten free"]),
        ("world cuisines", ["italian", "french", "mexican", "indian", "asian", "chinese", "japanese", "thai", "mediterranean", "african", "spanish"]),
        ("beverages", ["wine", "cocktail", "beer", "coffee", "tea", "drink", "juice"]),
        ("food science and technique", ["science", "technique", "professional", "chef", "culinary", "ferment", "preserv", "butcher"]),
        ("quick and practical cooking", ["quick", "easy", "simple", "minute", "instant pot", "slow cooker", "weeknight", "one-pot", "one pot"]),
    ]
    for topic, keywords in rules:
        if any(keyword in value for keyword in keywords):
            return topic
    return "general cooking"


def _number(text: str) -> float | None:
    match = re.search(r"([\d,.]+)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def scrape_goodreads(pages: int = 5, delay_seconds: float = 1.2) -> pd.DataFrame:
    """Scrape the public Best Cookbooks list, normally 100 books per page."""
    from bs4 import BeautifulSoup
    rows: list[dict[str, object]] = []
    for page in range(1, pages + 1):
        response = requests.get(
            GOODREADS_URL.format(page=page), headers=HEADERS, timeout=30
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        book_rows = soup.select('tr[itemtype="http://schema.org/Book"]')
        if not book_rows:
            raise RuntimeError(f"No books found on Goodreads page {page}; selectors may have changed.")

        for item in book_rows:
            title_node = item.select_one("a.bookTitle")
            author_node = item.select_one("a.authorName")
            rating_node = item.select_one("span.minirating")
            rank_node = item.select_one("td.number")
            cover_node = item.select_one("img.bookCover")
            if not title_node or not author_node:
                continue
            rating_text = rating_node.get_text(" ", strip=True) if rating_node else ""
            rating_match = re.search(r"([\d.]+) avg rating", rating_text)
            count_match = re.search(r"([\d,]+) ratings", rating_text)
            title = title_node.get_text(" ", strip=True)
            topic = infer_food_topic(title)
            rows.append(
                {
                    "title": title,
                    "authors": author_node.get_text(" ", strip=True),
                    "average_rating": float(rating_match.group(1)) if rating_match else None,
                    "rating_count": int(count_match.group(1).replace(",", "")) if count_match else None,
                    "first_publish_year": None,
                    "edition_count": None,
                    "subjects": f"{topic} | cookbooks | cooking | food and drink",
                    "languages": "eng",
                    "cover_url": cover_node.get("src") if cover_node else None,
                    "book_url": urljoin("https://www.goodreads.com", title_node.get("href", "")),
                    "source": "Goodreads web scraping",
                    "source_topic": topic,
                    "source_rank": int(_number(rank_node.get_text(strip=True)) or len(rows) + 1) if rank_node else len(rows) + 1,
                }
            )
        time.sleep(delay_seconds)

    return pd.DataFrame(rows).drop_duplicates(subset=["title", "authors"]).reset_index(drop=True)


def _join(values: object, limit: int = 30) -> str:
    return " | ".join(str(x).strip() for x in values[:limit]) if isinstance(values, list) else ""


def collect_open_library(
    target: int = 700,
    books_per_topic: int = 150,
    delay_seconds: float = 1.05,
) -> pd.DataFrame:
    """Collect food-related metadata in a few batched Open Library requests."""
    rows_by_topic: dict[str, list[dict[str, object]]] = {}
    for topic in FOOD_TOPICS:
        response = requests.get(
            OPEN_LIBRARY_URL,
            params={"subject": topic, "fields": OPEN_LIBRARY_FIELDS, "limit": books_per_topic},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        topic_rows: list[dict[str, object]] = []
        for book in response.json().get("docs", []):
            title = str(book.get("title", "")).strip()
            authors = _join(book.get("author_name", []), 10)
            subjects = _join(book.get("subject", []), 30)
            if not title or not authors or not subjects:
                continue
            cover_id = book.get("cover_i")
            work_key = book.get("key")
            topic_rows.append(
                {
                    "title": title,
                    "authors": authors,
                    "average_rating": book.get("ratings_average"),
                    "rating_count": book.get("ratings_count"),
                    "first_publish_year": book.get("first_publish_year"),
                    "edition_count": book.get("edition_count"),
                    "subjects": subjects,
                    "languages": _join(book.get("language", []), 10),
                    "cover_url": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None,
                    "book_url": f"https://openlibrary.org{work_key}" if work_key else None,
                    "source": "Open Library API",
                    "source_topic": topic,
                    "source_rank": None,
                }
            )
        rows_by_topic[topic] = topic_rows
        print(f"API topic complete: {topic} ({len(topic_rows)} valid records)", flush=True)
        time.sleep(delay_seconds)

    # Round-robin selection prevents the early topics from dominating when the
    # global target is lower than the number of valid API responses.
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    max_rows = max((len(rows) for rows in rows_by_topic.values()), default=0)
    for rank in range(max_rows):
        for topic in FOOD_TOPICS:
            topic_rows = rows_by_topic.get(topic, [])
            if rank >= len(topic_rows):
                continue
            row = topic_rows[rank]
            key = re.sub(r"\W+", "", f"{row['title']}{row['authors']}".lower())
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= target:
                return pd.DataFrame(selected).reset_index(drop=True)
    if len(selected) < target:
        raise RuntimeError(f"Only {len(selected)} unique API books were available; target was {target}.")
    return pd.DataFrame(selected).reset_index(drop=True)


def collect_all(output_dir: str = "data", reuse_scraped: bool = False) -> dict[str, int]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    scraped_path = directory / "books_scraped.csv"
    scraped = pd.read_csv(scraped_path) if reuse_scraped and scraped_path.exists() else scrape_goodreads()
    api = collect_open_library()
    combined = pd.concat([scraped, api], ignore_index=True)
    combined["dedupe_key"] = (
        combined["title"].str.lower().str.replace(r"\W+", "", regex=True)
        + combined["authors"].str.lower().str.replace(r"\W+", "", regex=True)
    )
    combined = combined.drop_duplicates("dedupe_key").drop(columns="dedupe_key").reset_index(drop=True)
    scraped.to_csv(directory / "books_scraped.csv", index=False, encoding="utf-8-sig")
    api.to_csv(directory / "books_api.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(directory / "books_combined.csv", index=False, encoding="utf-8-sig")
    return {"scraped": len(scraped), "api": len(api), "combined_unique": len(combined)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--reuse-scraped", action="store_true",
        help="Reuse the cached Goodreads CSV while refreshing the API records.",
    )
    args = parser.parse_args()
    print(collect_all(args.output_dir, reuse_scraped=args.reuse_scraped))


if __name__ == "__main__":
    main()

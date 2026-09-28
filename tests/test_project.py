import pandas as pd
import json
from pathlib import Path
from scipy import sparse

from src.recommender import find_book, normalize, recommend_books
from src.train_model import clean_books


def sample_books():
    books = pd.DataFrame([
        {"title": "Food Safety", "authors": "N. Potter", "subjects": "food hygiene | food safety", "languages": "eng", "source_topic": "food safety", "source": "test", "average_rating": 4.0, "rating_count": 100, "first_publish_year": 1986, "edition_count": 4, "cluster": 0, "known_domain": "Food Safety"},
        {"title": "Safe Food Handling", "authors": "Harold McGee", "subjects": "food hygiene | safety", "languages": "eng", "source_topic": "food safety", "source": "test", "average_rating": 4.5, "rating_count": 500, "first_publish_year": 1984, "edition_count": 10, "cluster": 0, "known_domain": "Food Safety"},
        {"title": "Nutrition Basics", "authors": "A. Writer", "subjects": "nutrition | health", "languages": "eng", "source_topic": "nutrition", "source": "test", "average_rating": 4.1, "rating_count": 80, "first_publish_year": 2018, "edition_count": 2, "cluster": 1, "known_domain": "Nutrition"},
    ])
    matrix = sparse.csr_matrix([[1, 1, 0], [0.9, 1, 0], [0, 0, 1]])
    return books, matrix


def test_normalize_accents_and_punctuation():
    assert normalize("Café: Food!") == "cafe food"


def test_clean_books_builds_nlp_text():
    books, _ = sample_books()
    result = clean_books(books, balance=False)
    assert len(result) == 3
    assert "book_text" in result


def test_find_exact_and_typo_suggestion():
    books, _ = sample_books()
    assert find_book("Food Safety", books)[0] == 0
    assert "Food Safety" in find_book("Food Safet", books)[1]


def test_recommendation_remains_in_cluster():
    books, matrix = sample_books()
    result = recommend_books("Food Safety", books, matrix)
    assert result["found"]
    assert result["recommendations"][0]["title"] == "Safe Food Handling"


def test_submission_book_count_and_model_size():
    combined = pd.read_csv("data/books_combined.csv")
    clustered = pd.read_csv("data/clustered_books.csv")
    metrics = json.loads(Path("models/metrics.json").read_text(encoding="utf-8"))
    assert len(combined) >= 1050
    assert not combined.duplicated(["title", "authors"]).any()
    assert len(clustered) == len(combined)
    assert metrics["books_modelled"] == 600
    assert metrics["books_in_app"] == len(clustered)
    assert metrics["silhouette"] >= 0.50

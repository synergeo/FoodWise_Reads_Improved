from pathlib import Path

import pandas as pd
import streamlit as st
from scipy import sparse

from src.recommender import recommend_books


st.set_page_config(page_title="Food Book Recommender", page_icon="📚", layout="wide")
st.title("📚 FoodWise Reads")
st.write("Explore the full food-book catalogue using cluster-assisted content similarity.")
st.caption("The clustering metrics were measured on a balanced six-domain sample; all collected books can be selected here.")

catalog_path = Path("data/clustered_books.csv")
matrix_path = Path("models/recommendation_matrix.npz")
if not catalog_path.exists() or not matrix_path.exists():
    st.error("The prepared catalogue or model is missing. Run the collection and training commands in README.md.")
    st.stop()

books = pd.read_csv(catalog_path)
matrix = sparse.load_npz(matrix_path)
if matrix.shape[0] != len(books):
    st.error("Catalogue and recommendation matrix have different book counts. Retrain the model.")
    st.stop()
labels = (books["title"] + " — " + books["authors"]).sort_values().tolist()

with st.sidebar:
    st.header("Recommendation settings")
    number = st.slider("Number of recommendations", 3, 10, 5)
    topic = st.selectbox(
        "Optional domain filter",
        ["All food domains"] + sorted(books["known_domain"].dropna().unique().tolist()),
    )

choice = st.selectbox("Choose a food-related book", labels, index=None, placeholder="Start typing a title or author")
if choice and st.button("Recommend books", type="primary"):
    title, author = choice.split(" — ", 1)
    result = recommend_books(title, books, matrix, author, number, topic)
    st.caption(f"Selected content cluster: {result['cluster']}")
    if not result["recommendations"]:
        st.warning("No recommendations matched this cluster and filter. Try All food domains.")
    else:
        for book in result["recommendations"]:
            left, right = st.columns([1, 5])
            with left:
                if book["cover_url"]:
                    st.image(book["cover_url"], width=120)
            with right:
                st.subheader(book["title"])
                st.write(book["authors"])
                st.caption(f"Domain: {book['known_domain']}")
                details = [f"Similarity: {book['similarity']:.1%}"]
                if book["average_rating"] is not None:
                    details.append(f"Rating: {book['average_rating']}/5")
                if book["year"]:
                    details.append(f"First published: {book['year']}")
                st.caption(" · ".join(details))
                st.write(str(book["subjects"])[:300])
                if book["book_url"]:
                    st.link_button("View book source", book["book_url"])
            st.divider()

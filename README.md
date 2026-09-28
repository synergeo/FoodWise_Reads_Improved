# FoodWise Reads: Improved Food Book Recommender

An Ironhack two-week project combining web scraping, APIs, cleaning, EDA, NLP, K-Means, PCA, SQL, content-based similarity and Streamlit deployment.

## Business idea

The system helps food professionals, students and interested readers discover books across six focused domains: Baking and Pastry, Nutrition, Food Safety, Agriculture, Fermentation and Food Business.

## Improved clustering result

The original 992-book exploratory model achieved a silhouette score of 0.1113 because many food topics overlapped. The expanded project now retains **1,196 unique books** after cross-source deduplication, exceeding the 1,050-book submission target. A balanced, quality-controlled catalogue of **600 books** (100 per domain) is used for clustering and recommendations.

| Metric | Improved result |
|---|---:|
| Silhouette score | **0.6093** |
| Adjusted Rand Index | **0.6665** |
| Normalized Mutual Information | **0.6667** |
| Cluster sizes | 81-125 books |
| Stability across five seeds | 0.6093 every run |

The known domain and source topic are held out of the clustering features. The exact collection-topic phrase is also removed from subject text before TF-IDF, reducing label leakage.

## Assignment coverage

| Requirement | Implementation |
|---|---|
| Approximately 500 scraped books | Five pages of the public Goodreads Best Cookbooks list |
| At least 1,050 unique books | 500 scraped records plus 700 API records produced 1,196 unique books after deduplication |
| Approximately 700 API books | Balanced, round-robin food-domain requests to the Open Library Search API |
| Cleaning and EDA | Deduplication, type conversion, missing-value handling and summary outputs |
| NLP | Title and cleaned subject metadata transformed with TF-IDF unigrams/bigrams |
| Unsupervised learning | SVD representation testing followed by K-Means at the six-domain product design |
| PCA | Two-dimensional cluster coordinates saved for visualization |
| Recommender | Same-cluster candidates ranked by cosine similarity |
| SQL | Clean catalogue loaded into SQLite with indexes and example queries |
| Streamlit | Searchable book selector, topic filter, covers, similarity and source links |

## Install and run on Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.collect_data
python -m src.train_model
python -m src.database
streamlit run app.py
```

The app normally opens at `http://localhost:8501`.

## Recommendation logic

1. Find the selected title and author.
2. Read the cluster assigned by the improved TF-IDF + SVD + K-Means model.
3. Keep only books in the same cluster.
4. Apply the optional food-topic filter.
5. Calculate cosine similarity from the fitted feature matrix.
6. Return the closest books, excluding the selected title.

The cluster supplies a broad thematic neighbourhood. Cosine similarity creates the final ranking inside that neighbourhood.

## Data responsibility

The collector requests five public list pages slowly and identifies itself. API calls are batched by subject and cached locally in CSV files. Website structures and access rules can change; check the source terms and robots policy before every fresh collection. Open Library advises low-volume human-facing use and local caching; for large-scale use, use its data dumps.

## Limitations

- Goodreads list metadata is less detailed than Open Library metadata.
- Metadata similarity does not prove that an individual reader will like a book.
- The balanced modelling catalogue uses 600 of the 1,196 collected records; the remainder stays available for acquisition and EDA evidence.
- Sparse or inconsistent subject labels can still affect individual recommendations.
- K-Means simplifies books that naturally belong to several topics.
- Ratings from different sources are not perfectly comparable.

## Next steps

- Add reader profiles and collaborative filtering.
- Evaluate click, save and rating behaviour through an A/B test.
- Add German-language filters and professional levels.
- Introduce explainable labels for each cluster from its strongest TF-IDF terms.
- Replace live collection with scheduled, versioned datasets for reliability.

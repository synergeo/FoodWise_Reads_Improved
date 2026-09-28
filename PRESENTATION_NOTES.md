# FoodWise Reads - presentation notes

## 1. Problem and audience

Food professionals and interested readers face thousands of books across cooking, nutrition, food science, safety, agriculture and sustainability. The prototype recommends related books from metadata rather than requiring existing user ratings.

## 2. Data acquisition

- 500 books scraped from a public Goodreads cookbook list.
- 700 food-domain records requested through the Open Library Search API.
- 1,196 unique books remained after cross-source title-author deduplication, exceeding the 1,050-book target.
- Requests were delayed and results cached locally.

## 3. Cleaning and EDA

Titles and authors were required, and repeated records were removed using normalized title-author keys. The full acquisition catalogue contains 1,196 books. For modelling, six clearer food domains were consolidated and balanced at 100 books each.

## 4. NLP and clustering

TF-IDF generated unigram and bigram features from titles and cleaned subject metadata. The known domain and source-topic fields were excluded. The exact collection-topic phrase was removed from subject text before modelling. Truncated SVD reduced noise, and K-Means created six clusters.

## 5. Model result

The selected six-dimensional representation achieved a silhouette score of 0.6093, exceeding the 0.50 target. ARI was 0.6665 and NMI was 0.6667 against held-out domains. Cluster sizes ranged from 81 to 125 books. Five random seeds reproduced the same result.

## 6. Recommendation method

After matching the selected title and author, the program restricts candidates to the same K-Means cluster. It then ranks those candidates by cosine similarity and optionally applies a food-domain filter.

## 7. Engineering and deployment

The cleaned catalogue is stored in SQLite with title and cluster indexes. Streamlit provides a searchable selector, domain filter, book covers, similarity scores and source links.

## 8. Limitations and next steps

Metadata similarity does not represent personal taste. Subject quality varies between sources, and 600 of the 1,196 collected records enter the balanced modelling catalogue. A future hybrid system should combine content similarity with reader behaviour and evaluate clicks, saves and ratings.

## Closing line

“FoodWise Reads turns 1,196 unique books from two sources into a focused food discovery tool, with a stable silhouette score of 0.6093 and an interactive recommendation app.”

## 10. Thank you and questions

Thank the audience and invite questions about the data collection, leakage-controlled clustering, evaluation metrics and Streamlit recommender.

# Movie‑Tag Recommender (SBERT)

Tiny Python pipeline that embeds movie metadata with a SentenceTransformer and returns cosine‑similar recommendations.

## Requirements

- Python ≥ 3.9
- `sentence‑transformers`
- `scikit‑learn`
- `pandas`, `numpy`

```bash
pip install sentence-transformers scikit-learn pandas numpy
```

## Data

Expect a `DataFrame` with columns: `movieId, title, genres, year, tags`  
`tags` should already be a single comma‑separated string per row (see regex clean‑up step).

## Quick start

```python
vecs = build_vectors(csv_data)          # saves movie_vecs.npy
recs = recommend(csv_data, vecs, "space adventure", k=5)
print(recs)
```

## Key functions

| Function purpose                | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| `_embed(text)`                  | SBERT embedding, L2‑normalised                        |
| `_row_vec(row)`                 | Weighted sum of column embeddings (`COL_WEIGHTS`)     |
| `build_vectors(df)`             | Stack all row vectors → `np.ndarray` (saved to *.npy) |
| `recommend(df, vecs, query, k)` | Return top‑*k* similar movies                         |

*Set **`COL_WEIGHTS = {"tags":0.5, "genres":0.3, "tags":0.15, "year": 0.05}`** as needed.*

## Fine Tunning

- Change `COL_WEIGHTS` for if you want to build a recommendation system that values a column more than other, for instance you may think that "year" is far more relevant than only 0.05.

## Output

Returns original DataFrame slice with the highest cosine similarity scores.

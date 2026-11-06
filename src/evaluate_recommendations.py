
import numpy as np
import pandas as pd
import faiss
from pathlib import Path

# Load trained retrieval embeddings
user_embeddings = np.load("models/user_embeddings.npy")
movie_embeddings = np.load("models/movie_embeddings.npy")
user_ids = np.load("models/user_ids.npy").astype(str)
movie_ids = np.load("models/movie_ids.npy").astype(str)

# Load the exact training sample used to train retrieval
train = pd.read_csv(
    "data/processed/train_sample.csv",
    usecols=["userId", "movieId"]
)

# Load held-out validation ratings
validation = pd.read_csv(
    "data/processed/validation.csv",
    usecols=["userId", "movieId", "rating"]
)

train["userId"] = train["userId"].astype(str)
train["movieId"] = train["movieId"].astype(str)
validation["userId"] = validation["userId"].astype(str)
validation["movieId"] = validation["movieId"].astype(str)

# Remove unknown ID at index 0
valid_movies = movie_ids[1:]
vectors = movie_embeddings[1:].astype("float32")

# Build FAISS index
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

user_lookup = {uid: i for i, uid in enumerate(user_ids)}
known_movies = set(valid_movies)

# Movies already seen during training
seen = train.groupby("userId")["movieId"].agg(set).to_dict()

# Relevant held-out movies: rating >= 4
relevant = (
    validation[
        (validation["rating"] >= 4.0)
        & validation["movieId"].isin(known_movies)
    ]
    .groupby("userId")["movieId"]
    .agg(set)
    .to_dict()
)

# Only evaluate users with known embeddings and relevant movies
eligible = sorted(
    set(relevant) & (set(user_lookup) - {user_ids[0]})
)

# Keep runtime manageable
rng = np.random.default_rng(42)
if len(eligible) > 100:
    eligible = rng.choice(
        eligible, size=100, replace=False
    ).tolist()

precision_scores = []
recall_scores = []

for uid in eligible:
    user_vector = user_embeddings[
        user_lookup[uid]
    ].astype("float32").reshape(1, -1)

    # Search all indexed movies, then exclude seen movies
    scores, indices = index.search(
        user_vector, index.ntotal
    )

    recommendations = []
    seen_movies = seen.get(uid, set())

    for idx in indices[0]:
        movie = valid_movies[idx]

        if movie not in seen_movies:
            recommendations.append(movie)

        if len(recommendations) == 10:
            break

    relevant_movies = relevant[uid] - seen_movies
    if not relevant_movies:
        continue

    hits = len(set(recommendations) & relevant_movies)

    precision_scores.append(hits / 10)
    recall_scores.append(hits / len(relevant_movies))

if not precision_scores:
    raise ValueError(
        "No eligible users with unseen relevant movies."
    )

precision = float(np.mean(precision_scores))
recall = float(np.mean(recall_scores))

print("\nRECOMMENDATION EVALUATION")
print("-------------------------")
print("Users evaluated:", len(precision_scores))
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")

Path("outputs").mkdir(exist_ok=True)

pd.DataFrame([
    {"metric": "Precision@10", "value": precision},
    {"metric": "Recall@10", "value": recall}
]).to_csv(
    "outputs/recommendation_evaluation.csv",
    index=False
)

print("\nResults saved successfully!")

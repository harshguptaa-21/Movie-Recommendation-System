
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import faiss
import tensorflow as tf

# Load Stage 1 embeddings
user_embeddings = np.load("models/user_embeddings.npy")
movie_embeddings = np.load("models/movie_embeddings.npy")

user_ids = np.load("models/user_ids.npy").astype(str)
movie_ids = np.load("models/movie_ids.npy").astype(str)

# Load Stage 2 ranking model
ranking_model = tf.keras.models.load_model(
    "models/ranking_model.keras",
    compile=False
)

# Exclude unknown movie ID at index 0
valid_movie_embeddings = movie_embeddings[1:].astype("float32")
valid_movie_ids = movie_ids[1:]

# Create FAISS index
index = faiss.IndexFlatIP(valid_movie_embeddings.shape[1])
index.add(valid_movie_embeddings)

# Select a known user
user_id = str(user_ids[1])
user_matches = np.where(user_ids == user_id)[0]

if len(user_matches) == 0:
    raise ValueError("User not found in training vocabulary")

user_index = int(user_matches[0])

# Find movies already rated in training
ratings = pd.read_csv(
    "data/processed/train.csv",
    usecols=["userId", "movieId"]
)

watched_movies = set(
    ratings.loc[
        ratings["userId"] == int(user_id),
        "movieId"
    ].tolist()
)

# Stage 1: Retrieve candidates with FAISS
user_vector = user_embeddings[user_index].astype("float32")
user_vector = user_vector.reshape(1, -1)

scores, indices = index.search(user_vector, index.ntotal)

candidate_ids = []
retrieval_scores = []

for idx, score in zip(indices[0], scores[0]):
    if idx < 0:
        continue

    movie_id = int(valid_movie_ids[idx])

    if movie_id in watched_movies:
        continue

    candidate_ids.append(movie_id)
    retrieval_scores.append(float(score))

    if len(candidate_ids) == 100:
        break

if not candidate_ids:
    raise ValueError("No unseen candidate movies found")

# Stage 2: Convert candidates to model vocabulary indices
movie_index_map = {
    int(mid): i for i, mid in enumerate(movie_ids[1:], start=1)
}

candidate_indices = np.array(
    [movie_index_map[mid] for mid in candidate_ids],
    dtype=np.int32
).reshape(-1, 1)

user_indices = np.full(
    (len(candidate_ids), 1),
    user_index,
    dtype=np.int32
)

# Predict user preference
predictions = ranking_model.predict(
    {
        "user": user_indices,
        "movie": candidate_indices
    },
    verbose=0
).flatten()

# Load movie titles
movies = pd.read_csv("data/ml-25m/movies.csv")
movie_titles = dict(zip(movies["movieId"], movies["title"]))

# Sort candidates by predicted rating
results = []

for mid, score, predicted_rating in zip(
    candidate_ids, retrieval_scores, predictions
):
    results.append({
        "movie_id": mid,
        "title": movie_titles.get(mid, "Unknown Movie"),
        "retrieval_score": score,
        "predicted_rating": float(predicted_rating)
    })

results.sort(
    key=lambda item: item["predicted_rating"],
    reverse=True
)

print("\nTWO-STAGE RECOMMENDATION ENGINE")
print("User:", user_id)
print("Already rated:", len(watched_movies))
print("Candidates retrieved:", len(candidate_ids))
print("\nTOP 10 RECOMMENDATIONS:\n")

for rank, item in enumerate(results[:10], start=1):
    print(
        f"{rank}. {item['title']} | "
        f"Predicted rating: {item['predicted_rating']:.2f}/5"
    )

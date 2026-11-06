
import numpy as np
import pandas as pd
import faiss

# Load trained embeddings and IDs
user_embeddings = np.load("models/user_embeddings.npy")
movie_embeddings = np.load("models/movie_embeddings.npy")

user_ids = np.load("models/user_ids.npy")
movie_ids = np.load("models/movie_ids.npy")

# Exclude lookup entry 0 (unknown ID)
valid_movie_embeddings = movie_embeddings[1:].astype("float32")
valid_movie_ids = movie_ids[1:]

# Build FAISS index using dot-product similarity
dimension = valid_movie_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(valid_movie_embeddings)

print("Movies indexed:", index.ntotal)

# Select an existing user
user_id = str(user_ids[1])
print("Selected user:", user_id)

matches = np.where(user_ids == user_id)[0]

if len(matches) == 0:
    raise ValueError(f"User {user_id} is not in the training sample.")

# Retrieve user embedding
user_vector = user_embeddings[matches[0]].astype("float32")
user_vector = user_vector.reshape(1, -1)

# Load training ratings
ratings = pd.read_csv(
    "data/processed/train.csv",
    usecols=["userId", "movieId"]
)

# Find movies already rated by this user
watched_movies = set(
    ratings.loc[
        ratings["userId"] == int(user_id),
        "movieId"
    ].tolist()
)

print("Movies already rated:", len(watched_movies))

# Search all indexed movies, then exclude watched ones
scores, indices = index.search(user_vector, index.ntotal)

# Load movie titles
movies = pd.read_csv("data/ml-25m/movies.csv")
movie_titles = dict(zip(movies["movieId"], movies["title"]))

print(f"\nTop 10 unseen movie candidates for User {user_id}:\n")

rank = 0

for idx, score in zip(indices[0], scores[0]):
    if idx < 0:
        continue

    movie_id = int(valid_movie_ids[idx])

    # Skip movies already rated by the user
    if movie_id in watched_movies:
        continue

    title = movie_titles.get(movie_id, "Unknown Movie")

    rank += 1
    print(f"{rank}. {title} | Score: {score:.4f}")

    if rank == 10:
        break

if rank == 0:
    print("No unseen movies available in the index.")

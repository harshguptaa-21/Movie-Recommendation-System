
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import setuptools
import distutils # type: ignore

import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained ranking model
model = tf.keras.models.load_model(
    "models/ranking_model.keras",
    compile=False
)

# Load the saved training vocabularies
user_ids = np.load("models/user_ids.npy").astype(str)
movie_ids = np.load("models/movie_ids.npy").astype(str)

# Create mappings from actual IDs to embedding indices
user_map = {
    uid: i for i, uid in enumerate(user_ids)
}

movie_map = {
    mid: i for i, mid in enumerate(movie_ids)
}

# Load chronological validation data
validation = pd.read_csv(
    "data/processed/validation.csv",
    usecols=["userId", "movieId", "rating"]
)

print("Total validation ratings:", len(validation))

# Convert IDs to strings
validation["userId"] = validation["userId"].astype(str)
validation["movieId"] = validation["movieId"].astype(str)

# Keep only users and movies known to the model
validation = validation[
    validation["userId"].isin(user_map) &
    validation["movieId"].isin(movie_map)
].copy()

print("Known user/movie validation ratings:", len(validation))

if validation.empty:
    raise ValueError(
        "No validation ratings contain both known users and movies."
    )

# Convert IDs to embedding indices
user_indices = validation["userId"].map(user_map).to_numpy(
    dtype=np.int32
).reshape(-1, 1)

movie_indices = validation["movieId"].map(movie_map).to_numpy(
    dtype=np.int32
).reshape(-1, 1)

actual_ratings = validation["rating"].to_numpy(
    dtype=np.float32
)

# Predict ratings
predictions = model.predict(
    {
        "user": user_indices,
        "movie": movie_indices
    },
    batch_size=1024,
    verbose=0
).flatten()

# Calculate evaluation metrics
mae = np.mean(np.abs(actual_ratings - predictions))

rmse = np.sqrt(
    np.mean((actual_ratings - predictions) ** 2)
)

print("\nRANKING MODEL EVALUATION")
print("------------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save results
os.makedirs("outputs", exist_ok=True)

results = pd.DataFrame([{
    "metric": "MAE",
    "value": float(mae)
}, {
    "metric": "RMSE",
    "value": float(rmse)
}])

results.to_csv(
    "outputs/ranking_evaluation.csv",
    index=False
)

print("\nResults saved to outputs/ranking_evaluation.csv")

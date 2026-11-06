
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import setuptools
import distutils # type: ignore

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the same training sample used for retrieval
ratings = pd.read_csv(
    "data/processed/train_sample.csv",
    usecols=["userId", "movieId", "rating"]
)


# Load the exact vocabularies from Stage 1
user_ids = np.load("models/user_ids.npy").astype(str)
movie_ids = np.load("models/movie_ids.npy").astype(str)

# Build lookup layers using saved vocabularies
# The first entry is reserved for unknown IDs
user_lookup = tf.keras.layers.StringLookup(
    vocabulary=user_ids[1:].tolist(),
    mask_token=None
)

movie_lookup = tf.keras.layers.StringLookup(
    vocabulary=movie_ids[1:].tolist(),
    mask_token=None
)

# Prepare inputs
users = ratings["userId"].astype(str).to_numpy()
movies = ratings["movieId"].astype(str).to_numpy()
targets = ratings["rating"].to_numpy(dtype="float32")

# Convert string IDs into integer indices
user_indices = user_lookup(tf.constant(users)).numpy()
movie_indices = movie_lookup(tf.constant(movies)).numpy()

# Define input layers
user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="user")
movie_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="movie")

# Deep component: learn user/movie representations
user_embedding = tf.keras.layers.Embedding(
    len(user_ids), 32
)(user_input)

movie_embedding = tf.keras.layers.Embedding(
    len(movie_ids), 32
)(movie_input)

user_vector = tf.keras.layers.Flatten()(user_embedding)
movie_vector = tf.keras.layers.Flatten()(movie_embedding)

combined = tf.keras.layers.Concatenate()([
    user_vector, movie_vector
])

deep = tf.keras.layers.Dense(64, activation="relu")(combined)
deep = tf.keras.layers.Dense(32, activation="relu")(deep)

# Wide component: linear combination of embeddings
wide = tf.keras.layers.Dense(1)(combined)

# Combine wide and deep components
merged = tf.keras.layers.Concatenate()([wide, deep])

# Predict rating between 0.5 and 5.0
output = tf.keras.layers.Dense(1, activation="sigmoid")(merged)
output = tf.keras.layers.Rescaling(scale=4.5, offset=0.5)(output)

model = tf.keras.Model(
    inputs=[user_input, movie_input],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("Starting ranking model training...")

model.fit(
    {"user": user_indices, "movie": movie_indices},
    targets,
    batch_size=256,
    epochs=5,
    validation_split=0.1
)

# Save the ranking model
Path("models").mkdir(exist_ok=True)

model.save("models/ranking_model.keras")

print("Ranking model training completed!")
print("Model saved to models/ranking_model.keras")

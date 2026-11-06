
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Load a sample of the training data
ratings = pd.read_csv(
    "data/processed/train_sample.csv",
    usecols=["userId", "movieId"]
)

# Convert IDs into strings
user_ids = ratings["userId"].astype(str).to_numpy()
movie_ids = ratings["movieId"].astype(str).to_numpy()

# Create lookup layers for users and movies
user_lookup = tf.keras.layers.StringLookup(
    vocabulary=np.unique(user_ids),
    mask_token=None
)

movie_lookup = tf.keras.layers.StringLookup(
    vocabulary=np.unique(movie_ids),
    mask_token=None
)

# Create embeddings
embedding_size = 32

user_model = tf.keras.Sequential([
    user_lookup,
    tf.keras.layers.Embedding(
        user_lookup.vocabulary_size(),
        embedding_size
    )
])

movie_model = tf.keras.Sequential([
    movie_lookup,
    tf.keras.layers.Embedding(
        movie_lookup.vocabulary_size(),
        embedding_size
    )
])

# Define a retrieval model
class RetrievalModel(tfrs.Model):
    def __init__(self):
        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"])

        return self.task(
            user_embeddings,
            movie_embeddings
        )

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices({
    "userId": user_ids,
    "movieId": movie_ids
})

dataset = dataset.shuffle(100000).batch(256)

# Train the model
model = RetrievalModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

print("Starting retrieval model training...")

model.fit(dataset, epochs=3)

print("Retrieval model training completed!")


from pathlib import Path

# Create a folder for saved model files
Path("models").mkdir(exist_ok=True)

# Extract learned embeddings
user_embeddings = model.user_model.layers[1].get_weights()[0]
movie_embeddings = model.movie_model.layers[1].get_weights()[0]

# Save embeddings and their corresponding IDs
np.save("models/user_embeddings.npy", user_embeddings)
np.save("models/movie_embeddings.npy", movie_embeddings)

np.save(
    "models/user_ids.npy",
    np.array(user_lookup.get_vocabulary())
)

np.save(
    "models/movie_ids.npy",
    np.array(movie_lookup.get_vocabulary())
)

print("Embeddings saved successfully!")
print("User embeddings shape:", user_embeddings.shape)
print("Movie embeddings shape:", movie_embeddings.shape)

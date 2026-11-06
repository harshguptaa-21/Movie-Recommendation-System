
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import setuptools
import distutils # type: ignore

import streamlit as st
import numpy as np
import pandas as pd
import faiss
import tensorflow as tf

st.set_page_config(page_title="Movie Recommendation System")

st.title("Movie Recommendation System")
st.write("A simple movie recommendation project using Machine Learning.")


# Load models and data only once
@st.cache_resource
def load_models():
    user_embeddings = np.load("models/user_embeddings.npy")
    movie_embeddings = np.load("models/movie_embeddings.npy")

    user_ids = np.load("models/user_ids.npy").astype(str)
    movie_ids = np.load("models/movie_ids.npy").astype(str)

    ranking_model = tf.keras.models.load_model(
        "models/ranking_model.keras",
        compile=False
    )

    index = faiss.IndexFlatIP(movie_embeddings.shape[1])
    index.add(movie_embeddings[1:].astype("float32"))

    return user_embeddings, user_ids, movie_ids, ranking_model, index


@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "data/processed/train.csv",
        usecols=["userId", "movieId"]
    )

    movies = pd.read_csv("data/ml-25m/movies.csv")
    titles = dict(zip(movies["movieId"], movies["title"]))

    return ratings, titles


user_embeddings, user_ids, movie_ids, ranking_model, index = load_models()
ratings, movie_titles = load_data()

st.subheader("Get Movie Recommendations")

user_id = st.text_input("Enter User ID", value=str(user_ids[1]))

if st.button("Recommend Movies"):

    matches = np.where(user_ids == user_id)[0]

    if len(matches) == 0 or matches[0] == 0:
        st.error("User ID not found. Please enter a valid training user ID.")

    else:
        user_index = int(matches[0])

        watched = set(
            ratings.loc[
                ratings["userId"].astype(str) == user_id,
                "movieId"
            ].tolist()
        )

        user_vector = user_embeddings[user_index].astype("float32")
        user_vector = user_vector.reshape(1, -1)

        scores, indices = index.search(user_vector, index.ntotal)

        candidate_ids = []

        for idx in indices[0]:
            if idx < 0:
                continue

            movie_id = int(movie_ids[idx + 1])

            if movie_id not in watched:
                candidate_ids.append(movie_id)

            if len(candidate_ids) == 100:
                break

        if not candidate_ids:
            st.warning("No unseen movies found for this user.")

        else:
            movie_index_map = {
                int(mid): i
                for i, mid in enumerate(movie_ids[1:], start=1)
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

            predictions = ranking_model.predict(
                {
                    "user": user_indices,
                    "movie": candidate_indices
                },
                verbose=0
            ).flatten()

            results = []

            for mid, rating in zip(candidate_ids, predictions):
                results.append({
                    "Movie": movie_titles.get(mid, "Unknown Movie"),
                    "Predicted Rating": round(float(rating), 2)
                })

            results.sort(
                key=lambda x: x["Predicted Rating"],
                reverse=True
            )

            st.success("Recommendations generated successfully!")
            st.write("Movies already rated:", len(watched))
            st.write("Candidates retrieved:", len(candidate_ids))

            st.subheader("Top 10 Recommended Movies")

            st.dataframe(
                pd.DataFrame(results[:10]),
                hide_index=True
            )

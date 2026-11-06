# Movie Recommendation System

A machine learning project that recommends movies to users based on their past ratings. It uses a two-stage approach: first finding candidate movies, then predicting how much a user might like them.

## Features

- Recommends 10 movies for a selected user ID
- Filters out movies already rated by that user in the training data
- Uses FAISS to retrieve 100 candidate movies
- Predicts ratings using a TensorFlow ranking model
- Displays recommendations in a simple Streamlit web app

## Dataset

This project uses the **MovieLens 25M** dataset, which contains movie information and user ratings. The training scripts use a smaller sample of the data to make experimentation manageable on a personal laptop.

Dataset: https://grouplens.org/datasets/movielens/25m/

## How it works

1. **Prepare data:** Split the ratings data into training and validation sets.
2. **Retrieval:** Train user and movie embeddings with TensorFlow Recommenders.
3. **Candidate search:** Use FAISS to find similar movie embeddings for a user.
4. **Ranking:** Predict ratings for the retrieved movies using a TensorFlow model.
5. **Recommendations:** Remove movies the user already rated, rank the remaining candidates, and show the top 10.

## Technologies used

Python, pandas, NumPy, TensorFlow, TensorFlow Recommenders, FAISS, Streamlit, and PySpark (for feature-processing experiments).

## Project structure

```text
Recommendation-Engine/
├── app.py                      # Streamlit interface
├── src/                        # Data preparation, training, evaluation, recommendations
├── data/                       # Dataset files (not included in GitHub)
├── models/                     # Saved models and embeddings (not included in GitHub)
├── outputs/                    # Evaluation results
├── docs/                       # Project screenshots
├── requirements.txt
└── README.md
```

## Run the app locally

Create and activate a Python virtual environment, then install the project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and prepare the MovieLens data and train the models using the scripts in `src/` before starting the app. Then run:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`), enter a user ID from the training data, and click **Recommend Movies**.

> **Note:** The app needs the trained model and embedding files in `models/`. Those files are generated locally and are not included in this repository.

## Evaluation

| Metric | Result |
|---|---:|
| Ranking MAE | 0.7918 |
| Ranking RMSE | 1.0422 |
| Precision@10 | 0.0260 |
| Recall@10 | 0.0075 |

The ranking evaluation used validation ratings for known user/movie IDs. Recommendation evaluation used 100 eligible users. These results show that the system works, but recommendation accuracy has room for improvement.

## Screenshots

Add a screenshot of the running Streamlit app to `docs/` and link it here after saving it.

## Limitations

- The retrieval and ranking models are trained on sampled ratings rather than the full dataset.
- Recommendations work best for users and movies represented in the training sample.
- The current model does not use movie descriptions, posters, or detailed content features.
- The app runs locally and is intended as a learning project.

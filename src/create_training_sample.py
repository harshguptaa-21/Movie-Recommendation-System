
import pandas as pd
from pathlib import Path

# Read the existing chronological training split
train = pd.read_csv(
    "data/processed/train.csv",
    usecols=["userId", "movieId", "rating", "timestamp"]
)

print("Total training ratings:", len(train))

# Sample across the entire training period
sample = train.sample(
    n=500000,
    random_state=42
)

# Sort the sampled interactions chronologically
sample = sample.sort_values("timestamp")

# Save separately so the original training data stays intact
Path("data/processed").mkdir(parents=True, exist_ok=True)

sample.to_csv(
    "data/processed/train_sample.csv",
    index=False
)

print("Sampled ratings:", len(sample))
print("Unique users:", sample["userId"].nunique())
print("Unique movies:", sample["movieId"].nunique())
print("Training sample created successfully!")

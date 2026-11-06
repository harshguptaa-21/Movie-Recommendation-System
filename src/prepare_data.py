
import pandas as pd
from pathlib import Path

# Load the ratings dataset
ratings = pd.read_csv("data/ml-25m/ratings.csv")

print("Total ratings:", len(ratings))

# Sort by timestamp (oldest first)
ratings = ratings.sort_values("timestamp")

# Calculate split positions
train_end = int(len(ratings) * 0.8)
val_end = int(len(ratings) * 0.9)

# Split without shuffling
train = ratings.iloc[:train_end]
val = ratings.iloc[train_end:val_end]
test = ratings.iloc[val_end:]

# Create output folder
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save the datasets
train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/validation.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Training ratings:", len(train))
print("Validation ratings:", len(val))
print("Testing ratings:", len(test))
print("Chronological split completed!")

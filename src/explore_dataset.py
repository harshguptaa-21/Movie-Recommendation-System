
import pandas as pd

# Load movie information
movies = pd.read_csv("data/ml-25m/movies.csv")

# Load user ratings
ratings = pd.read_csv("data/ml-25m/ratings.csv")

# Display the first five movies
print("FIRST 5 MOVIES:")
print(movies.head())

# Display the first five ratings
print("\nFIRST 5 RATINGS:")
print(ratings.head())

# Dataset information
print("\nTOTAL MOVIES:", len(movies))
print("TOTAL RATINGS:", len(ratings))
print("TOTAL USERS:", ratings["userId"].nunique())

# Check missing values
print("\nMISSING VALUES:")
print(ratings.isnull().sum())

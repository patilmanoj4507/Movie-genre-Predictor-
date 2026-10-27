
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "dataset/wiki_movie_plots_deduped.csv"
df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully!")
print("Columns:", df.columns.tolist())

# Select required columns
df = df[["Plot", "Genre"]].dropna().copy()

# Clean text and labels
df["Plot"] = df["Plot"].astype(str).str.strip()
df["Genre"] = df["Genre"].astype(str).str.strip().str.lower()

df = df[
    (df["Plot"] != "") &
    (df["Genre"] != "") &
    (df["Genre"] != "unknown")
].copy()

# Remove genres with too few examples for a reliable split
genre_counts = df["Genre"].value_counts()
valid_genres = genre_counts[genre_counts >= 5].index
df = df[df["Genre"].isin(valid_genres)].copy()

# Keep the 10 most common genres
top_genres = df["Genre"].value_counts().head(10).index
df = df[df["Genre"].isin(top_genres)].copy()

print("\nTotal movies used:", len(df))
print("\nGenre counts:")
print(df["Genre"].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["Plot"],
    df["Genre"],
    test_size=0.2,
    random_state=42,
    stratify=df["Genre"]
)

# Build TF-IDF + Logistic Regression model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nModel accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Save complete pipeline
joblib.dump(model, "movie_genre_model.pkl")

print("\nSUCCESS: Model saved as movie_genre_model.pkl")

import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import random

# Step 1: Load Data from SQL Database
def load_data(db_path):
    """Load data from the SQL database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT name, article FROM songs"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# Step 2: Extract TF-IDF Features
def extract_features(texts, max_features=5000):
    """Extract TF-IDF features from the text."""
    tfidf = TfidfVectorizer(max_features=max_features)
    return tfidf.fit_transform(texts).toarray()

# Step 3: Generate Synthetic Recommendation Levels
def generate_recommendation_levels(num_songs):
    """Simulate recommendation levels for training."""
    return np.random.rand(num_songs)

# Step 4: Create Training Samples
def create_training_samples(features, num_samples=500, num_songs=500):
    """Create synthetic training samples."""
    X_train, y_train = [], []
    for _ in range(num_samples):
        selected_indices = random.sample(range(len(features)), num_songs)
        selected_features = features[selected_indices]
        recommendations = generate_recommendation_levels(num_songs)
        X_train.append(selected_features)
        y_train.append(recommendations)
    return np.array(X_train), np.array(y_train)

# Step 5: Train the Model
def train_model(features):
    """Train the Random Forest model."""
    X_train, y_train = create_training_samples(features)
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    y_train_flat = y_train.flatten()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_flat, y_train_flat)
    return model

# Step 6: Save the Model
def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

# Step 7: Load the Model
def load_model(filename):
    """Load a trained model from a file."""
    return joblib.load(filename)

# Step 8: Recommend Songs
def recommend_songs(model, song_indices, article_features, data, top_k=10):
    """Recommend top K songs based on the model's predictions."""
    selected_features = article_features[song_indices]
    predictions = model.predict(selected_features)
    ranked_indices = np.argsort(-predictions)  # Descending order
    return data.iloc[song_indices].iloc[ranked_indices[:top_k]]['name'].values

# Main Program
if __name__ == "__main__":
    # Load data
    data = load_data('songs.db')
    print("Data loaded successfully!")

    # Extract features
    article_features = extract_features(data['article'])
    print("Features extracted successfully!")

    # Train or load the model
    try:
        model = load_model('song_recommendation_model.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Training model...")
        model = train_model(article_features)
        save_model(model, 'song_recommendation_model.pkl')
        print("Model trained and saved successfully!")

    # Recommend songs
    test_songs = random.sample(range(len(data)), 500)
    recommendations = recommend_songs(model, test_songs, article_features, data)
    print("Top recommended songs:", recommendations)

import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import time
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Step 1: Load Data from SQL Database
def load_data(db_path):
    """Load data from the SQL database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT name, article FROM songs LIMIT 10000"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data  # pandas.DataFrame

def preprocess_data(data):
    """
    Tokenize the data, remove stop words, punctuation, apply stemming and lemmatization.

    Args:
        data (2d array): Array in the format [[title1, article1], [title2, article2], ...]
    """
    tokenized_data = []
    
    print("Tokenizing data...")
    # Tokenize titles and articles
    start = time.time()
    for index, row in data.iterrows():
        title, article = row['name'], row['article']
        tokenized_article = nltk.word_tokenize(article)
        tokenized_data.append([title, tokenized_article])
        
        if index%1000 == 0:
            print(f"Tokenization has been running for: {time.time() - start} seconds")
            print(f"{index/len(data) *100}% complete")
            print()
            
    
    # Process each song's title and article
    processed_data = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    print("Filtering, stemming, and lemmatizing data")
    start = time.time()
    for index, (title, article_tokens) in enumerate(tokenized_data):
        # Filter out stop words and punctuation for both title and article
        filtered_article = [word.lower() for word in article_tokens if word.lower() not in stop_words and word not in string.punctuation]
        
        # Apply stemming to both title and article
        stemmed_article = [stemmer.stem(word) for word in filtered_article]
        
        # Apply lemmatization to both title and article
        lemmatized_article = [lemmatizer.lemmatize(word, wordnet.VERB) for word in stemmed_article]
        
        # Add processed title and article to the final list
        processed_data.append([title, lemmatized_article])
        
        if index%1000 == 0:
            print(f"Processing for {time.time() - start} seconds")
            print(f"{index/len(tokenized_data) *100}% complete")
            print()

    print("Data preprocessed!")
    return processed_data

def preprocess_article(article):
    """
    Tokenize, remove stop words, punctuation, apply stemming and lemmatization to a single article.

    Args:
        article (str): The article to be preprocessed.
    """
    # Tokenize the article
    tokenized_article = nltk.word_tokenize(article)

    # Filter out stop words and punctuation
    filtered_article = [word.lower() for word in tokenized_article if word.lower() not in stop_words and word not in string.punctuation]

    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_article = [stemmer.stem(word) for word in filtered_article]

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_article = [lemmatizer.lemmatize(word, wordnet.VERB) for word in stemmed_article]

    return lemmatized_article


def vectorize_data(data, vectorizer):    
    titles = [item[0] for item in data]
    article_texts = [" ".join(item[1]) for item in data]
    
    article_vectors = vectorizer.fit_transform(article_texts)
    
    article_title_mapping = {titles[i]: article_vectors[i] for i in range(len(titles))}
    return article_title_mapping

def recommend_songs(user_article, article_title_mapping, vectorizer, top_n=10000):
    # Convert the user's article into a vector
    user_article = " ".join(user_article)
    user_vector = vectorizer.transform([user_article])
    
    # Compute cosine similarities between the user's vector and all article vectors
    similarities = cosine_similarity(user_vector, np.vstack([vec.toarray() for vec in article_title_mapping.values()]))
    
    # Flatten the similarity matrix to make it easier to work with
    similarities = similarities.flatten()
    
    # Get the indices of the top N most similar songs
    top_n_indices = similarities.argsort()[-top_n:][::-1]
    
    # Get the song titles and their similarity scores
    recommended_songs_with_scores = [(list(article_title_mapping.keys())[index], similarities[index]) for index in top_n_indices]
    
    # Create a dictionary to hold the songs grouped by similarity score
    grouped_by_similarity = {}
    
    for song, score in recommended_songs_with_scores:
        if score not in grouped_by_similarity:
            grouped_by_similarity[score] = []
        grouped_by_similarity[score].append((song, score))
    
    # Now, for each group of songs with the same similarity score, select the one with the longest title
    final_recommendations = []
    
    for score, songs in grouped_by_similarity.items():
        # Get the song with the longest title in this group
        longest_title_song = max(songs, key=lambda x: len(x[0]))
        final_recommendations.append(longest_title_song)
    
    # Sort the final recommendations by similarity score in descending order
    final_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return final_recommendations





db_path = "songs.db"
data = load_data(db_path)
vectorizer = TfidfVectorizer(stop_words=None)

processed_data = preprocess_data(data)

vectorized_data = vectorize_data(processed_data, vectorizer)

my_article = """"
    """


print("Recommending songs...")
start = time.time()
recommended_songs = recommend_songs(preprocess_article(my_article), vectorized_data, vectorizer)
print(f"{time.time()-start}")
for index, (song, score) in enumerate(recommended_songs):
    print(f"{index}. Song: {song.ljust(100)} Similarity: {score}")


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_lg')

def tokenize_article(article):
    return word_tokenize(article)

def preprocess_article(article):
    stop_words = set(stopwords.words('english'))
    tokenized = tokenize_article(article)
    filtered_words = [word for word in tokenized if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(filtered_words)

def get_entities(article):
    doc = nlp(article)
    entities = set([ent.text.lower() for ent in doc.ents])
    return entities

def compare_articles(article1, article2):
    entities1 = get_entities(article1)
    entities2 = get_entities(article2)

    common_entities = entities1.intersection(entities2)
    max_similarity = len(common_entities)/max(len(entities1), len(entities2))
    min_similarity = len(common_entities)/min(len(entities1), len(entities2))
    avg_similarity = 2*len(common_entities)/(len(entities1)+len(entities2))

    print(f"Max Similarity: {max_similarity}")
    print(f"Min Similarity: {min_similarity}")
    print(f"Avg Similarity: {avg_similarity}")

    doc1 = nlp(preprocess_article(article1))
    doc2 = nlp(preprocess_article(article2))
    doc_similarity = doc1.similarity(doc2)
    print(f"Doc Similarity: {doc_similarity}")


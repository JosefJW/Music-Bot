import json
import sqlite3
import time
import wikipedia

conn = sqlite3.connect('songs.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY,
    name TEXT,
    album TEXT,
    article TEXT
)
''')
conn.commit()


def load_songs(file_num):
    print(f"Loading file: {file_num}")
    try:
        with open('song_titles'+str(file_num)+'.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("song_titles"+str(file_num)+".json not found.")
        return []

def load_progress():
    try:
        with open('progress.json', 'r') as f:
            progress = json.load(f)
        return (progress.get("current_index", 0), progress.get("file_num", 1))
    except FileNotFoundError:
        return (0, 1)  # If no progress file exists, start from the beginning

def save_progress(current_index, current_file):
    with open('progress.json', 'w') as f:
        json.dump({"current_index": current_index, "file_num": current_file}, f)

def get_wikipedia_article(song_title):
    try:
        page = wikipedia.page(song_title)
        return page.content
    except Exception as e:
        return None

def get_articles(start_index=0, start_file=1):   
    for j in range(start_file, 12):
        file_num = j
        songs = load_songs(file_num)
        for i in range(start_index, len(songs)):
            song = songs[i]
            name = song['name']
            album = song['album']['name']
            identification = song['id']

            article = get_wikipedia_article(name)
            if not article:
                article = get_wikipedia_article(album)
                if not article:
                    save_progress(i+1, file_num)
                    continue
            
            if (name not in article and album not in article) or ("song" not in article):
                save_progress(i+1, file_num)
                continue

            try:
                cursor.execute('''
                INSERT OR IGNORE INTO songs (id, name, album, article) VALUES (?, ?, ?, ?)
                ''', (identification, name, album, article))
                conn.commit()
            except sqlite3.Error as e:
                print(f"Database error: {e}")
            save_progress(i+1, file_num)
        start_index = 0

# Load songs and progress
(start_index, start_file) = load_progress()  # Get the index to start from

get_articles(start_index=start_index, start_file = start_file)

conn.close()
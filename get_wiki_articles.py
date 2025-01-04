import json
import sqlite3
import time
import wikipedia

songs = []
file_num = 1
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



def load_songs():
    global songs
    try:
        with open('song_titles.json', 'r') as f:
            songs = json.load(f)
    except FileNotFoundError:
        print("song_titles.json not found.")
        songs = []

def load_progress():
    try:
        with open('progress.json', 'r') as f:
            progress = json.load(f)
        return progress.get("current_index", 0)
    except FileNotFoundError:
        return 0  # If no progress file exists, start from the beginning

def save_progress(current_index):
    with open('progress.json', 'w') as f:
        json.dump({"current_index": current_index}, f)

def get_wikipedia_article(song_title):
    try:
        page = wikipedia.page(song_title)
        return page.content
    except Exception as e:
        print("No article :(")
        return None

def get_articles(start_index=0):
    song_info = {}
    num = start_index + 1  # Adjust for index start
    for i in range(start_index, len(songs)):
        song = songs[i]
        print(str(num) + ": " + song['name'])
        print(song['album']['name'])
        if song['album']['name'] not in song_info.keys():
            song_info[song['album']['name']] = get_wikipedia_article(song['album']['name'])
        print()
        num += 1

        # Save progress after each song to allow resuming
        save_progress(i + 1)  # Save the next index as the current progress

    # Save the songs with features to a JSON file
    with open('songs.json', 'w') as f:
        json.dump(song_info, f, indent=4)

# Load songs and progress
load_songs()
start_index = load_progress()  # Get the index to start from

try:
    get_articles(start_index=start_index)
except KeyboardInterrupt:
    print("\nProgram interrupted. Saving progress...")
    save_progress(start_index)
    # Optionally, save the current state to 'songs.json' as well
    with open('songs.json', 'w') as f:
        json.dump({}, f, indent=4)  # Save empty or partial progress

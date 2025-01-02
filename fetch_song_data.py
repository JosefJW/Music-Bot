import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import time
import spotify_auth
import wikipedia

# Initialize Spotify client with authentication
sp = spotify_auth.get_spotify_client()

def get_all_songs():
    # Initialize a list to store all songs
    all_songs = []
    
    # Set the search query to find tracks
    limit = 50  # Number of tracks per request
    offset = 0  # Start with the first page
    max_offset = 1000  # Maximum allowed offset in one request
    
    while offset < max_offset:
        # Fetch search results
        results = sp.search(q=query, type="track", limit=limit, offset=offset)
        
        # Get the tracks from the results
        tracks = results['tracks']['items']
        
        # Add the tracks to the all_songs list
        all_songs.extend(tracks)
        
        # If there are no more results, break the loop
        if len(tracks) < limit:
            break
        
        # Otherwise, increase the offset to get the next page
        offset += limit
    
    return all_songs

def get_wikipedia_article(song_title):
    try:
        page = wikipedia.page(song_title)
        return page.content
    except Exception as e:
        print("No article :(")
        return None

def increment_char(char):
    ascii_value = ord(char)
    new_ascii = ascii_value+1
    return chr(new_ascii)

# Fetching all songs
query = "A"
for i in range(26):
    print("Query: "+query)
    songs = get_all_songs()
    query = increment_char(query)

print("Getting song info...")
song_info = {}
for song in songs:
    print(song['name'])
    print(song['album']['name'])
    if song['album']['name'] not in song_info.keys:
        song_info[song['album']['name']] = get_wikipedia_article(song['album']['name'])



# Save the songs with features to a JSON file
with open('songs.json', 'w') as f:
    json.dump(song_info, f, indent=4)

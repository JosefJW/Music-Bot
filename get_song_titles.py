import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import time
import spotify_auth
import search_queries_list

# Initialize Spotify client with authentication
sp = spotify_auth.get_spotify_client()
songs = []
search_queries = search_queries_list.search_queries
file_num = 0

def get_all_songs(search):
    # Initialize a list to store all songs
    all_songs = []
    
    # Set the search query to find tracks
    limit = 50  # Number of tracks per request
    offset = 0  # Start with the first page
    max_offset = 1000  # Maximum allowed offset in one request
    
    while offset < max_offset:
        # Fetch search results
        results = sp.search(q=search, type="track", limit=limit, offset=offset)
        
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

def increment_char(char):
    ascii_value = ord(char)
    new_ascii = ascii_value+1
    return chr(new_ascii)

def alphabet_search():
    # Fetching all songs
    query = "A"
    print("Query: "+query)
    songs = get_all_songs(query)
    for i in range(25):
        query = increment_char(query)
        print("Query: "+query)
        songs.extend(get_all_songs(query))
        
    song_titles = set([track['name'] for track in songs])  # Get unique titles
    unique_songs = [track for track in songs if track['name'] in song_titles]

    with open('song_titles.json', 'w') as f:
        json.dump(songs, f)

def search_query_search():
    global file_num
    for i in range(start, len(search_queries)):
        query = search_queries[i]
        print("Query: " + query)
        songs = get_all_songs(query)
        with open('song_titles'+str(file_num)+'.json', 'r') as f:
            try:
                existing_songs = json.load(f)
            except json.JSONDecodeError:
                existing_songs = []
        existing_songs.extend(songs)
        with open('song_titles'+str(file_num)+'.json', 'w') as f:
            json.dump(existing_songs, f)
        with open('song_titles_progress.json', 'w') as f:
            json.dump("i: "+ str(i) + " File: " + str(file_num), f)
        if i%100 == 0:
            file_num += 1
        #time.sleep(60)

start = 1112
file_num = 11
search_query_search()
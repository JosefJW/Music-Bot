import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="53f784119a994de783d44b3690dd4dd3",
    client_secret="45e8e4e4e5f2403fb310180a9835e554",
    redirect_uri="http://localhost:8080/callback",
    scope=[]
))

user = sp.current_user()
print(f"Logged in as: {user['display_name']}")

def get_spotify_client():
    return sp
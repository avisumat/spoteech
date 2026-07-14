#all the actions are performed here
import httpx
from fastapi import HTTPException
#this function simply and dumbly fetches results and sends it to our search node, where we can
#start normalizing the whole data
async def search_spotify_api(query: str, search_types: str, token: str) -> dict:
    """
    Makes a GET request to spotify /search endpoint.
    :param query: The optimized search query.
    :param search_types: A comma-separated string (e.g. "track,playlist").
    """
    url = "https://api.spotify.com/v1/search"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    params = {
        "q": query,
        "type": search_types, # We pass the raw string exactly as it comes from LangGraph
        "limit": 10
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url=url, headers=headers, params=params)

            if response.status_code == 401:
                raise HTTPException(status_code=401, detail="Spotify Token Expired")

            if response.status_code == 200:
                print("REQUEST SUCCESSFUL...")
                # Return the whole raw dictionary! Let LangGraph do the hard work.
                return response.json() 
            else:
                print(f"REQUEST FAILURE... with status code : {response.status_code}")
                return {}

        except Exception as e:
            print(f"NETWORK ERROR : {e}")
            return {}
    #we use response.json() when response is json
    #we use response.text() when response is raw text or HTML content


async def play_spotify_tracks(items: list[dict], token: str, intent: str, device_id: str | None = None):
    # this scope is covered by "user-modify-playback-state" Any reviewer must ensure that this is added in the scope list
    """
    :param token:access token which is provided after authentication(OAuth2.0)
    :param intent: based on the intent we will use the optional body params(uris/context_uri)
    :body
        ::context_uri -> refer to track_uri [USE IT WHEN YOU WANT TO PLAY A COLLECTION(ALBUM,PLAYLIST,ARTIST etc.)]
        ::uris -> JSON array of the Spotify track URIs to play [USE IT WHEN YOU WANT TO PLAY SPECIFIC TRACKS(ONE OR MORE)]
    :optional body param : offset -> indicates from where in the context playback should start.Only available when context
                                     corresponds to an album or playlist object "position" is zero based
    """

    # Explanation: We changed logic to use 'uris' for both intents because we are sending a list of Track IDs.
    # 'context_uri' is strictly for Albums/Playlists and causes a 400 Bad Request error when used with Track IDs.

    url = "https://api.spotify.com/v1/me/player/play"

    # before making the request extract the uri from items which is an array of TrackObject
    # important note here that songs obtained is a dictionary where items is a list so we need to iterate the list to get uris
    # each item of the list is a further dictionary

    header = {
        "Authorization": f"Bearer {token}",
        # "Content-Type":"application/json" [use if the going to use content param in PUT method which sends raw bytes]
    }
    # 1. Grab the Top Hit's URI
    if not items:
        print("Error: No items found to play.")
        return

    top_hit_uri = items[0]["uri"]
    track_uris = [item.get("uri") for item in items if item.get("uri", "").startswith("spotify:track:")]
    body = {}

    # 2. Route the payload structure based on the Intent
    if intent == "individual" or track_uris:
        # Tracks MUST be in an array under the "uris" key
        body = {
            "uris": track_uris[:50] if track_uris else [top_hit_uri]
        }
    else:
        # Playlists ("queue"), Albums, and Artists MUST be a single string under "context_uri"
        body = {
            "context_uri": top_hit_uri
        }
    params = {}
    if device_id:
        params["device_id"] = device_id

    async with httpx.AsyncClient() as client:
        try:
            if device_id:
                transfer_response = await client.put(
                    url="https://api.spotify.com/v1/me/player",
                    headers={**header, "Content-Type": "application/json"},
                    json={"device_ids": [device_id], "play": False}
                )
                if transfer_response.status_code not in (200, 204):
                    print(f"DEVICE TRANSFER FAILURE... with status code : {transfer_response.status_code}")

            response = await client.put(url=url, headers=header, params=params, json=body)

            # --- ERROR HANDLING START ---
            if response.status_code  == 401:
                # Raises error so main.py can trigger re-login
                raise HTTPException(status_code=401, detail="Spotify Token Expired")

            if response.status_code == 404:
                # Handles "No Active Device" error
                print("ERROR: No active device found. Please start playing Spotify manually on your device first.")
                return
                # -----------------------------

            print(f"{response.status_code}Playback started successfully")
        except HTTPException:
            raise
        except Exception as e:
            print(f"ERROR : {e}")
            print("please fallback and try to update again...")


async def fetch_user_playlists(access_token:str):
    """
    endpoint - @app.get("/playlists")
    function name - get_user_playlists
    file- main.py
    """
    #the name is different so as to not cause confusion
    #mapping is given here:
    playlist_url = "https://api.spotify.com/v1/me/playlists"
    headers={
        "Authorization":f"Bearer {access_token}"
    }
    async with httpx.AsyncClient() as client:
        resp= await client.get(url=playlist_url,headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.json())


async def fetch_recent_tracks(access_token:str):
    """
    endpoint - @app.get("/recent")
    function name - get_recent_tracks
    file- main.py
    """
    recent_url = "https://api.spotify.com/v1/me/player/recently-played"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    async with httpx.AsyncClient() as client:
        resp= await client.get(url=recent_url,headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise HTTPException(status_code=resp.status_code, detail=resp.json())
        

async def fetch_queue(access_token:str):
    """
    endpoint - @app.get("/api/queue")
    function name - get_queue
    file - main.py
    """
    async with httpx.AsyncClient() as client:
        res = await client.get("https://api.spotify.com/v1/me/player/queue", headers={"Authorization": f"Bearer {access_token}"})
        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=res.text)
        return res.json()
    
async def fetch_playback_state(access_token: str):
    """
    endpoint - @app.get("/api/state")
    file - main.py
    """
    state_url = "https://api.spotify.com/v1/me/player"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url=state_url, headers=headers)
        
        # 204 means Spotify is idle/nothing is playing
        if resp.status_code == 204: 
            return {} 
            
        if resp.status_code == 200:
            return resp.json()
            
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    

async def fetch_album_tracks(album_id: str, limit: int, offset: int, access_token: str) -> dict:
    spotify_id = album_id.split(":")[-1]
    url = f"https://api.spotify.com/v1/albums/{spotify_id}/tracks"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url=url, 
            headers={"Authorization": f"Bearer {access_token}"},
            params={"limit": limit, "offset": offset}
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code in (400, 403):
            raise HTTPException(status_code=response.status_code, detail=f"Spotify Access Error: {response.text}")
        elif response.status_code == 429:
            raise HTTPException(status_code=429, detail="Spotify Rate Limit Exceeded.")
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Spotify API Error: {response.text}")

async def fetch_playlist_tracks(playlist_id: str, limit: int, offset: int, access_token: str) -> dict:
    spotify_id = playlist_id.split(":")[-1]
    url = f"https://api.spotify.com/v1/playlists/{spotify_id}/tracks"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url=url, 
            headers={"Authorization": f"Bearer {access_token}"},
            params={"limit": limit, "offset": offset}
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code in (400, 403):
            raise HTTPException(status_code=response.status_code, detail=f"Spotify Access Error: {response.text}")
        elif response.status_code == 429:
            raise HTTPException(status_code=429, detail="Spotify Rate Limit Exceeded.")
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Spotify API Error: {response.text}")
        

async def fetch_user_top_items(item_type: str, access_token: str) -> dict:
    url = f"https://api.spotify.com/v1/me/top/{item_type}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"time_range": "short_term", "limit": 10}

    async with httpx.AsyncClient() as client:
        response = await client.get(url=url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=response.status_code, detail=response.text)

async def fetch_artist_albums(artist_id: str, access_token: str) -> dict:
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"include_groups": "album,single", "limit": 5}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url=url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail=response.text)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Could not reach Spotify servers.")
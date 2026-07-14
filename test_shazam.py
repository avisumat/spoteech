import asyncio
import logging
import pprint
from aiohttp_retry import ExponentialRetry
from shazamio import Shazam, Serialize, HTTPClient, SearchParams
from typing import Optional
from shazamio.schemas.models import SearchParams
from shazamio.utils import ExponentialRetry

logger =logging.getLogger(__name__)
logging.basicConfig(filename="test_shazam_io.log",
                    level = logging.DEBUG,
                    format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d - %(funcName)s()] - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    )


async def identify_audio(file_bytes: bytes) -> dict:
    shazam = Shazam(
        http_client=HTTPClient(
            retry_options=ExponentialRetry(
                attempts=5, max_timeout=100, statuses={500, 502, 503, 504, 429}
            )
        )
    )

    # 1. Recognize the track
    response = await shazam.recognize(file_bytes, options=SearchParams(segment_duration_seconds=7))
    
    if not response.get("matches"):
        return {"status": "error", "message": "Could not recognize the song. Please try again."}
    
    # 2. Serialize safely
    track_data = Serialize.full_track(response)
    
    if not track_data or not track_data.track:
        return {"status": "error", "message": "Song matched, but missing track metadata."}

    # 3. Crash-Proof Extraction
    title = getattr(track_data.track, "title", "Unknown Title")
    artist = getattr(track_data.track, "subtitle", "Unknown Artist")
    
    # Safely extract cover art without hardcoded indices
    images = getattr(track_data.track, "images", None)
    cover_image = getattr(images, "coverarthq", None) if images else None

    # Safely extract the Spotify Search Query string
    spotify_query = None
    # shazamio usually formats this as: spotify:search:track%3A<name>+artist%3A<artist>
    raw_spotify_url = getattr(track_data.track, "spotify_url", "")
    if raw_spotify_url and raw_spotify_url.startswith("spotify:search"):
        # Strip the "spotify:search:" prefix so the Spotify API accepts it properly
        spotify_query = raw_spotify_url.replace("spotify:search:", "")

    return {
        "status": "success",
        "title": title,
        "artist": artist,
        "spotify_search_query": spotify_query,
        "coverart": cover_image
    }
async def test_run():
    with open("Crumb Pit Just The Way It Goes.mp3","rb") as file:
        data_bytes= file.read()

    result= await identify_audio(data_bytes)
    print(f"FINAL DICTIONARY : {result}")

if __name__ == "__main__":
    asyncio.run(test_run())
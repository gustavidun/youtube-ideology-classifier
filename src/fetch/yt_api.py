import requests
from time import sleep
from itertools import islice
from youtube_transcript_api import YouTubeTranscriptApi
import time
import json
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Literal

from src.config import YT_API_KEYS

@dataclass
class Video():
    id : str
    title : Optional[str] = None
    channel : Optional[str] = None
    description : Optional[str] = None
    tags : Optional[list[str]] = None
    category : Optional[str] = None
    comments : Optional[list[str]] = None

    def is_metadata_empty(self) -> bool:
        attrs = (self.title, self.description, self.tags, self.comments)
        return all(a in (None, "", []) for a in attrs)

    def __str__(self):
        return f"Video ID: {self.id}"

VIDEOS_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/videos"
COMMENTS_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/commentThreads"
CHANNELS_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/channels"
PLAYLIST_ENDPOINT : str = "https://www.googleapis.com/youtube/v3/playlistItems"

MAX_API_ERRORS = 5

api_index : int = 0
error_counter : int = 0 


def request(url : str, params : dict):
    global api_index, error_counter
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        if resp:
            error_counter = 0
            return resp.json()
        else:
            print("No response.")
            return
    except requests.HTTPError as e:
        status = e.response.status_code
        err = json.loads(e.response.content)
        reason = err["error"]["errors"][0]["reason"]
        print(f"HTTP error... {reason}")
        if status == 403 and reason == "quotaExceeded":
            print(f"API index {api_index} reached quota.")
            if (api_index + 1 < len(YT_API_KEYS)):
                api_index += 1
                params.update({"key": YT_API_KEYS[api_index]})
                request(url, params)
            else:
                raise Exception("Quota exceeded.")
        if status == 400:
            print("Bad request.")
            return
    except (requests.Timeout, requests.ConnectionError):
        print("Connection error. Trying again in 5 seconds...")
        sleep(5)
        if error_counter <= MAX_API_ERRORS:
            request(url, params)
            error_counter += 1
        else:
            raise Exception("Max errors reached.")


def get_videos_metadata(vids : list[Video]) -> list[Video]:
    if len(vids) > 50: #split list 
        raise Exception("API can only process 50 videos at a time.")

    video_resp = request(VIDEOS_ENDPOINT, params = {
        "key": YT_API_KEYS[api_index],
        "part": "contentDetails, snippet, statistics",
        "id": ",".join([vid.id for vid in vids])
    })

    if not "items" in video_resp:
        print("No items in response")
        return
    
    vids = [
        Video(
            id=item["id"],
            channel=item["snippet"]["channelTitle"],
            title=item["snippet"]["title"],
            description=item["snippet"].get("description", None),
            category=item["snippet"].get("categoryId", None),
            tags=item["snippet"].get("tags", [])
        ) 
        for item in video_resp["items"]
    ]

    return vids


def get_channel_video_ids(channel_id: str) -> list[str]:
    channel_params = {
        "part": "contentDetails",
        "id": channel_id,
        "key": YT_API_KEYS[api_index]
    }
    
    channel_response = request(CHANNELS_ENDPOINT, params=channel_params)
    if not channel_response.get("items"):
        raise ValueError(f"Channel {channel_id} not found.")
        
    uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    
    video_ids = []
    next_page_token = None
    
    while True:
        playlist_params = {
            "part": "contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": 50,
            "key": YT_API_KEYS[api_index]
        }
        
        if next_page_token:
            playlist_params["pageToken"] = next_page_token
            
        playlist_response = request(PLAYLIST_ENDPOINT, params=playlist_params)
        
        for item in playlist_response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            
        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break
            
    return [Video(id=vid) for vid in video_ids]


def get_transcript(id : str, wait = 0):
    """Get transcript from video ID"""
    ytt_api = YouTubeTranscriptApi()
    if wait > 0: time.sleep(wait)

    try:
        t_dict = ytt_api.fetch(id).to_raw_data()
        print(f"Fetched transcript from video id {id}")
        transcript = [section["text"] for section in t_dict]
        return " ".join(transcript)
    except Exception as e:
        print(f"Transcript unavailable. Error {e}. Skipping.")
        return None


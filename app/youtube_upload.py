# youtube_upload.py
import os
import pickle
import google.auth.transport.requests
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time
import threading
import queue

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
CLIENT_SECRET_FILE = 'client_secret.json'
TOKEN_FILE = 'token.pickle'

def get_youtube_client():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(google.auth.transport.requests.Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(credentials, token)

    return build('youtube', 'v3', credentials=credentials)

upload_queue = queue.Queue()

def upload_to_youtube(file_path, title, description, tags, privacy='unlisted', playlist_id=None):
    youtube = get_youtube_client()
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags,
            'categoryId': '22'
        },
        'status': {
            'privacyStatus': privacy
        }
    }
    media = MediaFileUpload(file_path, chunksize=1024*1024, resumable=True)
    request = youtube.videos().insert(part='snippet,status', body=body, media_body=media)

    response = None
    attempt = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                logging.info(f"Upload progress: {progress}%")
        except Exception as e:
            if attempt < 10:
                wait = 2 ** attempt
                time.sleep(wait)
                attempt += 1
            else:
                raise e

    video_id = response['id']
    if playlist_id:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        ).execute()

    return f"https://youtu.be/{video_id}"

def process_upload_queue():
    while True:
        file_path, title, desc, tags, privacy, playlist = upload_queue.get()
        try:
            url = upload_to_youtube(file_path, title, desc, tags, privacy, playlist)
            logging.info(f"Uploaded: {url}")
        except Exception as e:
            logging.error(f"Upload failed: {e}")

upload_thread = threading.Thread(target=process_upload_queue, daemon=True)
upload_thread.start()

def start_upload(file_path, title, description, tags, privacy='unlisted', playlist_id=None):
    from utils import check_network_stability
    if not check_network_stability():
        raise ValueError("Network unstable")
    upload_queue.put((file_path, title, description, tags, privacy, playlist_id))

def debug_upload_to_youtube(file_path, *args):
    logging.info(f"Simulating upload of {file_path}")
    return "https://youtu.be/debug"
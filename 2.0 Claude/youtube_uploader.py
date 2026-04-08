"""
YouTube Uploader — uploads videos using the YouTube Data API v3.
Requires client_secrets.json in the app directory.
See README for setup instructions.
"""

import os
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class YouTubeUploader:
    def __init__(self):
        self.credentials_file = "client_secrets.json"
        self.token_file = "youtube_token.json"

    def upload_video(
        self,
        video_path: str,
        title: str,
        description: str,
        privacy: str = "private",
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Optional[str]:
        if not os.path.exists(self.credentials_file):
            logger.error("client_secrets.json not found — see README for YouTube API setup.")
            return None

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.auth.transport.requests import Request
            import pickle

            SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
            creds = None

            if os.path.exists(self.token_file):
                with open(self.token_file, "rb") as f:
                    creds = pickle.load(f)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                    creds = flow.run_local_server(port=0)
                with open(self.token_file, "wb") as f:
                    pickle.dump(creds, f)

            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": ["bird watching", "time-lapse", "nest box", "nature"],
                    "categoryId": "15",
                },
                "status": {"privacyStatus": privacy.lower()},
            }

            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status and progress_callback:
                    progress_callback(int(status.progress() * 100))

            video_id = response["id"]
            logger.info(f"Uploaded: https://www.youtube.com/watch?v={video_id}")
            return video_id

        except ImportError:
            logger.error("Install: pip install google-auth-oauthlib google-api-python-client")
            return None
        except Exception:
            logger.exception("Upload error")
            return None
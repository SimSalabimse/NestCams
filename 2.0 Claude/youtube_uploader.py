"""
YouTube Uploader Module
Handles uploading videos to YouTube using the YouTube Data API
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class YouTubeUploader:
    """
    YouTube uploader using Google API
    
    Requires:
    - Google API credentials (client_secrets.json)
    - OAuth 2.0 authentication
    
    Setup:
    1. Create a project in Google Cloud Console
    2. Enable YouTube Data API v3
    3. Create OAuth 2.0 credentials
    4. Download client_secrets.json to app directory
    """
    
    def __init__(self):
        self.credentials_file = 'client_secrets.json'
        self.token_file = 'youtube_token.json'
    
    def upload_video(self, video_path: str, title: str, description: str, 
                    privacy: str = 'private', progress_callback=None) -> Optional[str]:
        """
        Upload video to YouTube
        
        Args:
            video_path: Path to video file
            title: Video title
            description: Video description
            privacy: Privacy status (private, unlisted, public)
            progress_callback: Optional callback for upload progress
        
        Returns:
            Video ID if successful, None otherwise
        """
        
        if not os.path.exists(self.credentials_file):
            logger.error(f"Credentials file not found: {self.credentials_file}")
            logger.error("Please set up YouTube API credentials. See README for instructions.")
            return None
        
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.auth.transport.requests import Request
            import pickle
            
            # OAuth scopes
            SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
            
            credentials = None
            
            # Load saved credentials if available
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    credentials = pickle.load(token)
            
            # Refresh or get new credentials
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, SCOPES)
                    credentials = flow.run_local_server(port=0)
                
                # Save credentials for future use
                with open(self.token_file, 'wb') as token:
                    pickle.dump(credentials, token)
            
            # Build YouTube service
            youtube = build('youtube', 'v3', credentials=credentials)
            
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': ['bird watching', 'time-lapse', 'bird box', 'nature'],
                    'categoryId': '15'  # Pets & Animals
                },
                'status': {
                    'privacyStatus': privacy.lower()
                }
            }
            
            # Upload video
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = youtube.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status and progress_callback:
                    progress = int(status.progress() * 100)
                    progress_callback(progress)
            
            video_id = response['id']
            logger.info(f"Video uploaded successfully. Video ID: {video_id}")
            logger.info(f"URL: https://www.youtube.com/watch?v={video_id}")
            
            return video_id
            
        except ImportError:
            logger.error("Google API libraries not installed. Run: pip install google-auth-oauthlib google-api-python-client")
            return None
        except Exception as e:
            logger.exception("Error uploading to YouTube")
            return None

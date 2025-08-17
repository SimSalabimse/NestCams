from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from tkinter import messagebox
from utils import log_session, validate_video_file, check_network_stability
import time

def start_upload(app, file_path, task_name, button):
    """Start YouTube upload."""
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"File not found: {file_path}")
        return
    if not validate_video_file(file_path):
        messagebox.showerror("Error", "Invalid video file.")
        return
    if not check_network_stability():
        messagebox.showerror("Error", "Network unstable.")
        return
    button.configure(state="disabled", text="Uploading...")
    import threading
    threading.Thread(target=upload_to_youtube, args=(app, file_path, task_name, button)).start()
    log_session(f"Upload started for {file_path}")

def upload_to_youtube(app, file_path, task_name, button):
    """Upload video to YouTube."""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            youtube = get_youtube_client(app)
            if not youtube:
                messagebox.showerror("Error", "Failed to authenticate with YouTube.")
                break
            duration_str = task_name
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            title = file_name + (" #shorts" if "60s" in duration_str else "")
            description = "Uploaded via Bird Box Video Processor" + (" #shorts" if "60s" in duration_str else "")
            tags = ['bird', 'nature', 'video'] + (['#shorts'] if "60s" in duration_str else [])
            body = {
                'snippet': {'title': title, 'description': description, 'tags': tags, 'categoryId': '22'},
                'status': {'privacyStatus': 'unlisted'}
            }
            media = MediaFileUpload(file_path, resumable=True, chunksize=512 * 1024)
            request = youtube.videos().insert(part='snippet,status', body=body, media_body=media)
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = status.progress() * 100
                    app.queue.put(("upload_progress", file_path, progress))
            messagebox.showinfo("Success", f"Uploaded: https://youtu.be/{response['id']}")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                messagebox.showerror("Error", f"Upload failed: {str(e)}")
        finally:
            button.configure(state="normal", text="Upload to YouTube")

def debug_upload_to_youtube(app, file_path, task_name):
    """Simulate YouTube upload process."""
    start_time = time.time()
    log_session(f"Debug: Simulating YouTube upload for {file_path}")
    for i in range(0, 101, 10):
        app.queue.put(("upload_progress", file_path, i))
        time.sleep(0.05)  # Simulate upload progress
    log_session(f"Debug: YouTube upload simulated successfully")
    return f"Simulated upload for {task_name}: https://youtu.be/debug_{task_name}"

def get_youtube_client(app):
    """Authenticate YouTube API client."""
    import os
    if not hasattr(app, 'youtube_client'):
        credentials = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                credentials = pickle.load(token)
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secrets.json',
                    scopes=['https://www.googleapis.com/auth/youtube.upload']
                )
                credentials = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(credentials, token)
        app.youtube_client = build('youtube', 'v3', credentials=credentials)
    return app.youtube_client
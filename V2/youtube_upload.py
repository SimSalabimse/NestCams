"""
youtube_upload.py — YouTube Data API v3 uploader
Requires client_secrets.json in the program folder.
See Advanced tab for setup instructions.
"""

import os
import pickle
import threading
import logging
from tkinter import messagebox
from typing import Optional

from utils import log_session, validate_video_file, check_network_stability

logger = logging.getLogger(__name__)


def start_upload(app, file_path: str, task_name: str, button) -> None:
    """Validate, check network, then kick off an upload thread."""
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"File not found:\n{file_path}")
        return
    if not validate_video_file(file_path):
        messagebox.showerror("Error", "Video file appears corrupt.")
        return
    if not check_network_stability():
        messagebox.showerror("Error", "Network connection unstable.")
        return

    if button is not None:
        button.configure(state="disabled", text="Uploading…")

    threading.Thread(
        target=_upload_worker,
        args=(app, file_path, task_name, button),
        daemon=True,
    ).start()
    log_session(f"Upload started: {file_path}")
    logger.info(f"[UPLOAD] Upload thread started for {file_path}")


def _upload_worker(app, file_path: str, task_name: str, button) -> None:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            youtube = _get_client(app)
            if not youtube:
                messagebox.showerror("Error", "YouTube authentication failed.")
                break

            is_short   = "60s" in task_name
            base_name  = os.path.splitext(os.path.basename(file_path))[0]
            title       = base_name + (" #shorts" if is_short else "")
            description = "Uploaded via Bird Box Video Processor" + (" #shorts" if is_short else "")
            tags        = ["bird", "nestbox", "nature", "timelapse"] + (["#shorts"] if is_short else [])

            from googleapiclient.http import MediaFileUpload
            media   = MediaFileUpload(file_path, resumable=True, chunksize=512 * 1024)
            request = youtube.videos().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": title, "description": description,
                        "tags": tags, "categoryId": "15",
                    },
                    "status": {"privacyStatus": "unlisted"},
                },
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    pct = status.progress() * 100
                    app.queue.put(("upload_progress", file_path, pct))

            url = f"https://youtu.be/{response['id']}"
            log_session(f"Upload complete: {url}")
            logger.info(f"[UPLOAD] Complete: {url}")
            messagebox.showinfo("Upload Complete", f"Published at:\n{url}")
            break

        except Exception as exc:
            logger.warning(f"[UPLOAD] Attempt {attempt+1} failed: {exc}")
            if attempt == max_retries - 1:
                messagebox.showerror("Upload Failed", f"All {max_retries} attempts failed.\n{exc}")
        finally:
            if button is not None:
                try:
                    button.configure(state="normal", text="Upload to YouTube")
                except Exception:
                    pass


def _get_client(app):
    if hasattr(app, "_youtube_client"):
        return app._youtube_client

    secrets = "client_secrets.json"
    if not os.path.exists(secrets):
        messagebox.showerror(
            "Missing Credentials",
            "client_secrets.json not found.\n"
            "See the Advanced tab for YouTube setup instructions.",
        )
        return None

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
        creds  = None

        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as fh:
                creds = pickle.load(fh)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow  = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.pickle", "wb") as fh:
                pickle.dump(creds, fh)

        app._youtube_client = build("youtube", "v3", credentials=creds)
        log_session("YouTube client authenticated")
        return app._youtube_client

    except ImportError:
        messagebox.showerror(
            "Missing Package",
            "Install: pip install google-api-python-client google-auth-oauthlib",
        )
        return None
    except Exception as exc:
        logger.exception(f"[UPLOAD] Auth error: {exc}")
        messagebox.showerror("Auth Error", str(exc))
        return None

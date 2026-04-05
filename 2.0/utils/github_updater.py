import requests
from packaging import version
import json
from typing import Tuple
from .logger import logger


class GitHubUpdater:
    def __init__(self, repo: str = "yourusername/NestCams"):
        self.repo = repo
        self.api_url = f"https://api.github.com/repos/{repo}/releases/latest"

    def check_for_updates(self, current_version: str) -> Tuple[bool, str, str]:
        """Check if there's a newer version available"""
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            release_data = response.json()

            latest_version = release_data["tag_name"].lstrip("v")
            if version.parse(latest_version) > version.parse(current_version):
                return True, latest_version, release_data["html_url"]
            return False, current_version, ""
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return False, current_version, ""

    def get_release_notes(self, version: str) -> str:
        """Get release notes for a specific version"""
        try:
            response = requests.get(
                f"https://api.github.com/repos/{self.repo}/releases/tags/v{version}",
                timeout=10,
            )
            response.raise_for_status()
            release_data = response.json()
            return release_data.get("body", "No release notes available")
        except Exception as e:
            logger.error(f"Failed to get release notes: {e}")
            return "Unable to fetch release notes"

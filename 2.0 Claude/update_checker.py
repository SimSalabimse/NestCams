"""
Update Checker — checks GitHub for new releases.
"""

import requests
import logging
from typing import Tuple, Optional
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

CURRENT_VERSION = "2.0.0"
DEFAULT_GITHUB_REPO = "SimSalabimse/NestCams"


class UpdateChecker:
    def __init__(self, repo: Optional[str] = None, current_version: str = CURRENT_VERSION):
        self.current_version = current_version
        if repo:
            self.repo = repo
        else:
            self.repo = ConfigManager().get_setting("github_repo", DEFAULT_GITHUB_REPO)
        self.api_url = f"https://api.github.com/repos/{self.repo}/releases/latest"
        logger.info(f"UpdateChecker: repo={self.repo}")

    def get_repository(self) -> str:
        return self.repo

    def set_repository(self, repo: str) -> bool:
        if "/" not in repo or len(repo.split("/")) != 2:
            logger.error(f"Invalid repo format: {repo}")
            return False
        self.repo = repo
        self.api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        ConfigManager().set_setting("github_repo", repo)
        return True

    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        try:
            r = requests.get(self.api_url, timeout=10)
            if r.status_code == 404:
                return False, None, None
            r.raise_for_status()
            data = r.json()
            latest = data.get("tag_name", "").lstrip("v")
            url = data.get("html_url")
            has_update = self._is_newer(latest, self.current_version)
            return has_update, latest, url
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return False, None, None

    @staticmethod
    def _is_newer(latest: str, current: str) -> bool:
        try:
            lp = tuple(int(x) for x in latest.split("."))
            cp = tuple(int(x) for x in current.split("."))
            n = max(len(lp), len(cp))
            lp += (0,) * (n - len(lp))
            cp += (0,) * (n - len(cp))
            return lp > cp
        except Exception:
            return False
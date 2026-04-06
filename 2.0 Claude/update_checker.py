"""
Enhanced Update Checker Module
Checks for new releases on GitHub with configurable repository
"""

import requests
import logging
from typing import Tuple, Optional
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Application version
CURRENT_VERSION = "2.0.0"  # Updated with all new features

# Default GitHub repository (user can change in settings)
DEFAULT_GITHUB_REPO = "yourusername/bird-motion-processor"


class UpdateChecker:
    """Checks for application updates on GitHub"""
    
    def __init__(self, repo: Optional[str] = None, current_version: str = CURRENT_VERSION):
        self.current_version = current_version
        
        # Load repo from config or use provided/default
        if repo:
            self.repo = repo
        else:
            config_manager = ConfigManager()
            self.repo = config_manager.get_setting('github_repo', DEFAULT_GITHUB_REPO)
        
        self.api_url = f"https://api.github.com/repos/{self.repo}/releases/latest"
        logger.info(f"Update checker configured for repo: {self.repo}")
    
    def set_repository(self, repo: str) -> bool:
        """
        Set the GitHub repository to check
        
        Args:
            repo: Repository in format "owner/repo"
        
        Returns:
            True if valid format
        """
        if '/' not in repo or len(repo.split('/')) != 2:
            logger.error(f"Invalid repository format: {repo}. Use 'owner/repo'")
            return False
        
        self.repo = repo
        self.api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        
        # Save to config
        config_manager = ConfigManager()
        config_manager.set_setting('github_repo', repo)
        
        logger.info(f"Repository updated to: {repo}")
        return True
    
    def get_repository(self) -> str:
        """Get current repository"""
        return self.repo
    
    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check for updates on GitHub
        
        Returns:
            Tuple of (has_update, version, download_url)
        """
        try:
            logger.info(f"Checking for updates from: {self.api_url}")
            
            # Make request to GitHub API
            response = requests.get(self.api_url, timeout=10)
            
            if response.status_code == 404:
                logger.info("No releases found on GitHub")
                return False, None, None
            
            response.raise_for_status()
            
            data = response.json()
            latest_version = data.get('tag_name', '').lstrip('v')
            download_url = data.get('html_url')
            
            logger.info(f"Latest version on GitHub: {latest_version}")
            logger.info(f"Current version: {self.current_version}")
            
            # Compare versions
            has_update = self._is_newer_version(latest_version, self.current_version)
            
            if has_update:
                logger.info(f"Update available: {latest_version}")
            else:
                logger.info("Already on latest version")
            
            return has_update, latest_version, download_url
            
        except requests.RequestException as e:
            logger.error(f"Error checking for updates: {e}")
            return False, None, None
        except Exception as e:
            logger.exception("Unexpected error checking for updates")
            return False, None, None
    
    def _is_newer_version(self, latest: str, current: str) -> bool:
        """
        Compare version strings
        
        Args:
            latest: Latest version string (e.g., "1.2.0")
            current: Current version string (e.g., "1.0.0")
        
        Returns:
            True if latest is newer than current
        """
        try:
            # Parse versions as tuples of integers
            latest_parts = tuple(int(x) for x in latest.split('.'))
            current_parts = tuple(int(x) for x in current.split('.'))
            
            # Pad to same length
            max_len = max(len(latest_parts), len(current_parts))
            latest_parts = latest_parts + (0,) * (max_len - len(latest_parts))
            current_parts = current_parts + (0,) * (max_len - len(current_parts))
            
            return latest_parts > current_parts
            
        except (ValueError, AttributeError):
            # If parsing fails, assume no update
            logger.warning(f"Could not parse versions: {latest} vs {current}")
            return False
    
    def download_update(self, download_url: str, destination: str = "update.zip") -> bool:
        """
        Download update from GitHub
        
        Args:
            download_url: URL to download from
            destination: Local file path to save to
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Downloading update from: {download_url}")
            
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Update downloaded to: {destination}")
            return True
            
        except Exception as e:
            logger.exception("Error downloading update")
            return False

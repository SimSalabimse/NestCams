"""
Analytics service for processing statistics
"""

import logging

logger = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self, config):
        self.config = config

    def get_analytics(self):
        """Get analytics data"""
        return {"videos_processed": [], "avg_processing_time": 0, "success_rate": 1.0}

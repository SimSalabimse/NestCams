"""
Analytics service for processing statistics
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for collecting and analyzing processing statistics
    """

    def __init__(self, config):
        self.config = config
        self._stats: List[Dict[str, Any]] = []

    def record_processing(self, result: Any) -> None:
        """
        Record a processing result for analytics

        Args:
            result: ProcessingResult object
        """
        try:
            self._stats.append(
                {
                    "timestamp": (
                        result.timestamp if hasattr(result, "timestamp") else None
                    ),
                    "filename": result.filename,
                    "frames_processed": result.frames_processed,
                    "motion_events": result.motion_events,
                    "processing_time": result.processing_time,
                    "output_files": (
                        len(result.output_files) if result.output_files else 0
                    ),
                    "error": result.error,
                }
            )
        except AttributeError as e:
            logger.warning(f"Failed to record processing result: {e}")

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics data from recorded processing results

        Returns:
            Dictionary containing analytics data
        """
        if not self._stats:
            return {
                "videos_processed": [],
                "avg_processing_time": 0.0,
                "success_rate": 1.0,
                "total_videos": 0,
                "total_frames": 0,
            }

        successful = [stat for stat in self._stats if not stat.get("error")]
        success_rate = len(successful) / len(self._stats) if self._stats else 1.0

        processing_times = [
            stat["processing_time"]
            for stat in successful
            if stat.get("processing_time")
        ]
        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0.0
        )

        total_frames = sum(stat.get("frames_processed", 0) for stat in successful)

        return {
            "videos_processed": self._stats,
            "avg_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "total_videos": len(self._stats),
            "total_frames": total_frames,
        }

    def clear_stats(self) -> None:
        """Clear all recorded statistics"""
        self._stats.clear()
        logger.info("Analytics statistics cleared")

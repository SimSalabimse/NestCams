"""
Analytics Dashboard — tracks processing statistics in SQLite.
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    input_file TEXT,
                    video_duration REAL,
                    motion_duration REAL,
                    motion_percentage REAL,
                    segments_detected INTEGER,
                    processing_time REAL,
                    hw_acceleration TEXT
                )
            """)
            conn.commit()

    def log_processing(self, stats: Dict):
        try:
            motion = stats.get("motion_duration", 0)
            video = stats.get("video_duration", 1) or 1
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO processing_history
                    (timestamp, input_file, video_duration, motion_duration,
                     motion_percentage, segments_detected, processing_time, hw_acceleration)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    datetime.now().isoformat(),
                    stats.get("input_file", ""),
                    video,
                    motion,
                    motion / video * 100,
                    stats.get("motion_segments", 0),
                    stats.get("processing_time", 0),
                    stats.get("detection_method", "CPU"),
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Analytics log error: {e}")

    def get_summary(self, days: int = 30) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM processing_history").fetchone()[0]
                row = conn.execute("""
                    SELECT COUNT(*), SUM(motion_duration), AVG(motion_percentage), SUM(processing_time)
                    FROM processing_history
                    WHERE timestamp >= datetime('now', ? || ' days')
                """, (f"-{days}",)).fetchone()
        except Exception:
            return self._empty_summary()

        return {
            "total_videos_processed": total,
            "recent_videos": row[0] or 0,
            "recent_motion_time_hours": (row[1] or 0) / 3600,
            "average_motion_percentage": row[2] or 0,
            "total_processing_time_hours": (row[3] or 0) / 3600,
        }

    @staticmethod
    def _empty_summary() -> Dict:
        return {
            "total_videos_processed": 0,
            "recent_videos": 0,
            "recent_motion_time_hours": 0.0,
            "average_motion_percentage": 0.0,
            "total_processing_time_hours": 0.0,
        }

    def export_statistics(self, output_file: str):
        data = {
            "generated": datetime.now().isoformat(),
            "summary": self.get_summary(365),
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Statistics exported to {output_file}")
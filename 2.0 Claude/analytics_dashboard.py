"""
Analytics Dashboard Module
Track and visualize processing statistics
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List
import sqlite3

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """Track and analyze video processing statistics"""
    
    def __init__(self, db_path: str = 'analytics.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_file TEXT,
                video_duration REAL,
                motion_duration REAL,
                motion_percentage REAL,
                segments_detected INTEGER,
                frames_filtered INTEGER,
                filter_percentage REAL,
                target_length INTEGER,
                output_file TEXT,
                processing_time REAL,
                hw_acceleration TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frame_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                processing_id INTEGER,
                white_frames INTEGER,
                black_frames INTEGER,
                corrupted_frames INTEGER,
                blurry_frames INTEGER,
                total_frames INTEGER,
                FOREIGN KEY (processing_id) REFERENCES processing_history(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                videos_processed INTEGER,
                total_motion_time REAL,
                total_processing_time REAL,
                average_motion_percentage REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_processing(self, stats: Dict):
        """
        Log a processing session
        
        Args:
            stats: Dictionary with processing statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_history (
                timestamp, input_file, video_duration, motion_duration,
                motion_percentage, segments_detected, frames_filtered,
                filter_percentage, target_length, output_file,
                processing_time, hw_acceleration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            stats.get('input_file', ''),
            stats.get('video_duration', 0),
            stats.get('motion_duration', 0),
            stats.get('motion_percentage', 0),
            stats.get('segments_detected', 0),
            stats.get('frames_filtered', 0),
            stats.get('filter_percentage', 0),
            stats.get('target_length', 0),
            stats.get('output_file', ''),
            stats.get('processing_time', 0),
            stats.get('hw_acceleration', 'cpu')
        ))
        
        processing_id = cursor.lastrowid
        
        # Log frame statistics
        filter_stats = stats.get('filter_stats', {})
        cursor.execute('''
            INSERT INTO frame_statistics (
                processing_id, white_frames, black_frames,
                corrupted_frames, blurry_frames, total_frames
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            processing_id,
            filter_stats.get('white', 0),
            filter_stats.get('black', 0),
            filter_stats.get('corrupted', 0),
            filter_stats.get('blurry', 0),
            filter_stats.get('total', 0)
        ))
        
        conn.commit()
        conn.close()
        
        # Update daily summary
        self._update_daily_summary(stats)
    
    def _update_daily_summary(self, stats: Dict):
        """Update daily summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        # Check if entry exists
        cursor.execute('SELECT * FROM daily_summary WHERE date = ?', (today,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute('''
                UPDATE daily_summary SET
                    videos_processed = videos_processed + 1,
                    total_motion_time = total_motion_time + ?,
                    total_processing_time = total_processing_time + ?
                WHERE date = ?
            ''', (stats.get('motion_duration', 0), 
                  stats.get('processing_time', 0), today))
        else:
            # Create new
            cursor.execute('''
                INSERT INTO daily_summary (
                    date, videos_processed, total_motion_time,
                    total_processing_time, average_motion_percentage
                ) VALUES (?, 1, ?, ?, ?)
            ''', (today, stats.get('motion_duration', 0),
                  stats.get('processing_time', 0),
                  stats.get('motion_percentage', 0)))
        
        conn.commit()
        conn.close()
    
    def get_summary(self, days: int = 30) -> Dict:
        """
        Get summary statistics
        
        Args:
            days: Number of days to include
        
        Returns:
            Dictionary with summary stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total videos processed
        cursor.execute('SELECT COUNT(*) FROM processing_history')
        total_videos = cursor.fetchone()[0]
        
        # Recent activity (last N days)
        cursor.execute('''
            SELECT 
                COUNT(*),
                SUM(motion_duration),
                AVG(motion_percentage),
                SUM(processing_time)
            FROM processing_history
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        recent = cursor.fetchone()
        
        # Frame filtering stats
        cursor.execute('''
            SELECT 
                SUM(white_frames),
                SUM(black_frames),
                SUM(corrupted_frames),
                SUM(blurry_frames),
                SUM(total_frames)
            FROM frame_statistics
        ''')
        
        frames = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_videos_processed': total_videos,
            'recent_videos': recent[0] or 0,
            'recent_motion_time_hours': (recent[1] or 0) / 3600,
            'average_motion_percentage': recent[2] or 0,
            'total_processing_time_hours': (recent[3] or 0) / 3600,
            'frames_filtered': {
                'white': frames[0] or 0,
                'black': frames[1] or 0,
                'corrupted': frames[2] or 0,
                'blurry': frames[3] or 0,
                'total': frames[4] or 0
            }
        }
    
    def get_daily_stats(self, days: int = 7) -> List[Dict]:
        """Get daily statistics for last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM daily_summary
            WHERE date >= date('now', '-' || ? || ' days')
            ORDER BY date DESC
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = []
        for row in rows:
            stats.append({
                'date': row[0],
                'videos_processed': row[1],
                'total_motion_time': row[2],
                'total_processing_time': row[3],
                'average_motion_percentage': row[4]
            })
        
        return stats
    
    def export_statistics(self, output_file: str):
        """Export all statistics to JSON"""
        summary = self.get_summary(365)  # Last year
        daily = self.get_daily_stats(30)  # Last month
        
        export_data = {
            'generated': datetime.now().isoformat(),
            'summary': summary,
            'daily_stats': daily
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Statistics exported to: {output_file}")

"""
Database Management Module
=========================

SQLite database for analytics and logging with proper error handling.
"""

import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from models.data_models import DetectionResult

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database for analytics and logging"""
    
    def __init__(self, db_path: str = "indoor_nav.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Detections table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        object_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        distance REAL NOT NULL,
                        bbox TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        session_id TEXT NOT NULL
                    )
                ''')
                
                # Navigation sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        mode TEXT NOT NULL,
                        total_detections INTEGER DEFAULT 0,
                        total_distance REAL DEFAULT 0.0
                    )
                ''')
                
                # Analytics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        session_id TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def log_detection(self, detection: DetectionResult, session_id: str):
        """Log detection to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detections 
                    (timestamp, object_type, confidence, distance, bbox, risk_level, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection.timestamp.isoformat(),
                    detection.object_type,
                    detection.confidence,
                    detection.distance,
                    json.dumps(detection.bbox),
                    detection.risk_level,
                    session_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log detection: {e}")
    
    def log_session(self, session_id: str, mode: str, start_time: datetime):
        """Log new navigation session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (session_id, start_time, mode)
                    VALUES (?, ?, ?)
                ''', (
                    session_id,
                    start_time.isoformat(),
                    mode
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log session: {e}")
    
    def end_session(self, session_id: str, total_detections: int = 0, total_distance: float = 0.0):
        """End navigation session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET end_time = ?, total_detections = ?, total_distance = ?
                    WHERE session_id = ?
                ''', (
                    datetime.now().isoformat(),
                    total_detections,
                    total_distance,
                    session_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
    
    def get_analytics_data(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get detection counts by type
                cursor.execute('''
                    SELECT object_type, COUNT(*) as count
                    FROM detections
                    WHERE datetime(timestamp) >= datetime('now', '-{} days')
                    GROUP BY object_type
                    ORDER BY count DESC
                '''.format(days))
                
                detection_counts = dict(cursor.fetchall())
                
                # Get hourly activity
                cursor.execute('''
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                    FROM detections
                    WHERE datetime(timestamp) >= datetime('now', '-{} days')
                    GROUP BY hour
                    ORDER BY hour
                '''.format(days))
                
                hourly_activity = dict(cursor.fetchall())
                
                # Get recent sessions
                cursor.execute('''
                    SELECT session_id, start_time, mode, total_detections
                    FROM sessions
                    WHERE datetime(start_time) >= datetime('now', '-{} days')
                    ORDER BY start_time DESC
                    LIMIT 10
                '''.format(days))
                
                recent_sessions = cursor.fetchall()
                
                return {
                    'detection_counts': detection_counts,
                    'hourly_activity': hourly_activity,
                    'total_detections': sum(detection_counts.values()),
                    'most_common_object': max(detection_counts.items(), key=lambda x: x[1])[0] if detection_counts else None,
                    'recent_sessions': recent_sessions
                }
                
        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            return {
                'detection_counts': {},
                'hourly_activity': {},
                'total_detections': 0,
                'most_common_object': None,
                'recent_sessions': []
            }
    
    def export_data(self, filename: str = None) -> str:
        """Export database data to JSON file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analytics_export_{timestamp}.json"
            
            analytics_data = self.get_analytics_data()
            
            with open(filename, 'w') as f:
                json.dump(analytics_data, f, indent=2)
            
            logger.info(f"Data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return ""
    
    def clear_old_data(self, days: int = 30):
        """Clear old data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear old detections
                cursor.execute('''
                    DELETE FROM detections
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                '''.format(days))
                
                # Clear old sessions
                cursor.execute('''
                    DELETE FROM sessions
                    WHERE datetime(start_time) < datetime('now', '-{} days')
                '''.format(days))
                
                # Clear old analytics
                cursor.execute('''
                    DELETE FROM analytics
                    WHERE datetime(date) < datetime('now', '-{} days')
                '''.format(days))
                
                conn.commit()
                logger.info(f"Cleared data older than {days} days")
                
        except Exception as e:
            logger.error(f"Failed to clear old data: {e}") 
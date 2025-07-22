#!/usr/bin/env python3
"""
Database utilities for image recognition system
Handles RDS MySQL connections and data operations
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pymysql
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load database configuration from environment or config file"""
        # Try environment variables first (for production)
        if os.getenv('RDS_HOSTNAME'):
            return {
                'host': os.getenv('RDS_HOSTNAME'),
                'port': int(os.getenv('RDS_PORT', 3306)),
                'database': os.getenv('RDS_DB_NAME', 'image_recognition'),
                'username': os.getenv('RDS_USERNAME', 'admin'),
                'password': os.getenv('RDS_PASSWORD'),
                'charset': 'utf8mb4'
            }
        
        # Fallback to config file (for development)
        try:
            with open('/var/www/html/deployment-info.json', 'r') as f:
                deployment_config = json.load(f)
                return {
                    'host': deployment_config.get('rds_endpoint', 'localhost'),
                    'port': int(deployment_config.get('rds_port', 3306)),
                    'database': deployment_config.get('rds_database', 'image_recognition'),
                    'username': deployment_config.get('rds_username', 'admin'),
                    'password': deployment_config.get('rds_password', ''),
                    'charset': 'utf8mb4'
                }
        except Exception as e:
            logger.error(f"Failed to load database config: {e}")
            # Default configuration for development
            return {
                'host': 'localhost',
                'port': 3306,
                'database': 'image_recognition',
                'username': 'root',
                'password': '',
                'charset': 'utf8mb4'
            }

class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self):
        self.config = DatabaseConfig().config
        logger.info(f"Database config loaded: {self.config['host']}:{self.config['port']}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        connection = None
        try:
            connection = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['username'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset'],
                autocommit=False,
                cursorclass=pymysql.cursors.DictCursor
            )
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def create_image_record(self, s3_key: str, original_name: str, file_size: int) -> int:
        """Create new image record and return image ID"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                    INSERT INTO images (s3_key, original_name, file_size, processing_status)
                    VALUES (%s, %s, %s, 'pending')
                    """
                    cursor.execute(sql, (s3_key, original_name, file_size))
                    image_id = cursor.lastrowid
                    conn.commit()
                    logger.info(f"Created image record: ID={image_id}, S3={s3_key}")
                    return image_id
        except Exception as e:
            logger.error(f"Failed to create image record: {e}")
            raise
    
    def update_processing_status(self, image_id: int, status: str, processed_at: datetime = None):
        """Update image processing status"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if processed_at:
                        sql = """
                        UPDATE images 
                        SET processing_status = %s, processed_at = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """
                        cursor.execute(sql, (status, processed_at, image_id))
                    else:
                        sql = """
                        UPDATE images 
                        SET processing_status = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """
                        cursor.execute(sql, (status, image_id))
                    conn.commit()
                    logger.info(f"Updated image {image_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")
            raise
    
    def save_detection_results(self, image_id: int, rekognition_results: Dict):
        """Save Rekognition detection results to database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Save general labels
                    if 'labels' in rekognition_results:
                        for label in rekognition_results['labels']:
                            sql = """
                            INSERT INTO detection_labels (image_id, label_name, confidence)
                            VALUES (%s, %s, %s)
                            """
                            cursor.execute(sql, (image_id, label['Name'], label['Confidence']))
                    
                    # Save person detections
                    if 'boundingBoxes' in rekognition_results:
                        for person in rekognition_results['boundingBoxes']:
                            bbox = person['boundingBox']
                            sql = """
                            INSERT INTO person_detections 
                            (image_id, confidence, bbox_left, bbox_top, bbox_width, bbox_height)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """
                            cursor.execute(sql, (
                                image_id, person['confidence'],
                                bbox['Left'], bbox['Top'], bbox['Width'], bbox['Height']
                            ))
                    
                    # Save face detections
                    if 'faceBoxes' in rekognition_results:
                        for face in rekognition_results['faceBoxes']:
                            bbox = face['boundingBox']
                            
                            # Extract face attributes
                            age_low = face.get('ageRange', {}).get('Low')
                            age_high = face.get('ageRange', {}).get('High')
                            gender = face.get('gender', {}).get('Value')
                            gender_conf = face.get('gender', {}).get('Confidence')
                            
                            # Get primary emotion
                            emotions = face.get('emotions', [])
                            primary_emotion = emotions[0]['Type'] if emotions else None
                            emotion_conf = emotions[0]['Confidence'] if emotions else None
                            
                            # Insert face detection
                            sql = """
                            INSERT INTO face_detections 
                            (image_id, confidence, bbox_left, bbox_top, bbox_width, bbox_height,
                             age_low, age_high, gender, gender_confidence, 
                             primary_emotion, emotion_confidence)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            cursor.execute(sql, (
                                image_id, face['confidence'],
                                bbox['Left'], bbox['Top'], bbox['Width'], bbox['Height'],
                                age_low, age_high, gender, gender_conf,
                                primary_emotion, emotion_conf
                            ))
                            
                            face_id = cursor.lastrowid
                            
                            # Insert all emotions for this face
                            for emotion in emotions:
                                sql = """
                                INSERT INTO face_emotions (face_detection_id, emotion_type, confidence)
                                VALUES (%s, %s, %s)
                                """
                                cursor.execute(sql, (face_id, emotion['Type'], emotion['Confidence']))
                    
                    conn.commit()
                    logger.info(f"Saved detection results for image {image_id}")
                    
        except Exception as e:
            logger.error(f"Failed to save detection results: {e}")
            raise
    
    def get_image_by_s3_key(self, s3_key: str) -> Optional[Dict]:
        """Get image record by S3 key"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "SELECT * FROM images WHERE s3_key = %s"
                    cursor.execute(sql, (s3_key,))
                    return cursor.fetchone()
        except Exception as e:
            logger.error(f"Failed to get image by S3 key: {e}")
            return None
    
    def get_all_images_with_detections(self) -> List[Dict]:
        """Get all images with their detection results"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get basic image info with counts
                    sql = """
                    SELECT 
                        i.*,
                        COUNT(DISTINCT pd.id) as person_count,
                        COUNT(DISTINCT fd.id) as face_count,
                        COUNT(DISTINCT dl.id) as label_count
                    FROM images i
                    LEFT JOIN person_detections pd ON i.id = pd.image_id
                    LEFT JOIN face_detections fd ON i.id = fd.image_id  
                    LEFT JOIN detection_labels dl ON i.id = dl.image_id
                    GROUP BY i.id
                    ORDER BY i.upload_time DESC
                    """
                    cursor.execute(sql)
                    images = cursor.fetchall()
                    
                    # Get detailed detection data for each image
                    for image in images:
                        image_id = image['id']
                        
                        # Get labels
                        cursor.execute(
                            "SELECT label_name, confidence FROM detection_labels WHERE image_id = %s",
                            (image_id,)
                        )
                        image['labels'] = cursor.fetchall()
                        
                        # Get person detections
                        cursor.execute("""
                            SELECT confidence, bbox_left, bbox_top, bbox_width, bbox_height
                            FROM person_detections WHERE image_id = %s
                        """, (image_id,))
                        image['person_detections'] = cursor.fetchall()
                        
                        # Get face detections with emotions
                        cursor.execute("""
                            SELECT fd.*, GROUP_CONCAT(CONCAT(fe.emotion_type, ':', fe.confidence)) as emotions
                            FROM face_detections fd
                            LEFT JOIN face_emotions fe ON fd.id = fe.face_detection_id
                            WHERE fd.image_id = %s
                            GROUP BY fd.id
                        """, (image_id,))
                        image['face_detections'] = cursor.fetchall()
                    
                    return images
                    
        except Exception as e:
            logger.error(f"Failed to get images with detections: {e}")
            return []
    
    def log_processing_event(self, image_id: int, process_type: str, status: str, 
                           message: str = None, processing_time_ms: int = None):
        """Log processing events for monitoring"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                    INSERT INTO processing_logs 
                    (image_id, process_type, status, message, processing_time_ms)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (image_id, process_type, status, message, processing_time_ms))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to log processing event: {e}")

    def get_processing_status(self, image_id):
        """Get processing status for a specific image"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT processing_status, processed_at, upload_time
                        FROM images 
                        WHERE id = %s
                    """, (image_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        return {
                            'processing_status': result[0],
                            'processed_at': result[1],
                            'upload_time': result[2]
                        }
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return None

    def update_processing_status(self, image_id, status, processed_at=None):
        """Update processing status for an image"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if processed_at:
                        cursor.execute("""
                            UPDATE images 
                            SET processing_status = %s, processed_at = %s
                            WHERE id = %s
                        """, (status, processed_at, image_id))
                    else:
                        cursor.execute("""
                            UPDATE images 
                            SET processing_status = %s
                            WHERE id = %s
                        """, (status, image_id))
                    
                    conn.commit()
                    logger.info(f"Updated processing status for image {image_id}: {status}")
                    
        except Exception as e:
            logger.error(f"Error updating processing status: {e}")
            raise

# Global database manager instance
db_manager = DatabaseManager()

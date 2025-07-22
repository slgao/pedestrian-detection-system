#!/usr/bin/env python3
import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import boto3
from botocore.exceptions import ClientError
import logging

# Import database manager
try:
    from database import db_manager
    DATABASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Database module not available: {e}")
    DATABASE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients (removed rekognition_client - now handled by Lambda)
s3_client = boto3.client('s3', region_name='us-west-2')

# Load deployment configuration
def load_config():
    try:
        with open('/var/www/html/deployment-info.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            's3Bucket': 'my-app-image-bucket-20256200',
            'region': 'us-west-2'
        }

config = load_config()
S3_BUCKET = config.get('s3Bucket', 'my-app-image-bucket-20256200')

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """Upload images to S3 - Lambda will handle Rekognition processing asynchronously"""
    try:
        # Check for both 'files' (multiple) and 'file' (single) form fields
        files = []
        if 'files' in request.files:
            files = request.files.getlist('files')
        elif 'file' in request.files:
            files = request.files.getlist('file')
        else:
            return jsonify({'error': 'No files provided'}), 400
        
        logger.info(f"Received {len(files)} files for upload")
        uploaded_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"uploads/{uuid.uuid4()}{file_extension}"
            
            try:
                # Get file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                # Upload to S3 (this will trigger SNS → Lambda → Rekognition)
                s3_client.upload_fileobj(
                    file,
                    S3_BUCKET,
                    unique_filename,
                    ExtraArgs={
                        'ContentType': file.content_type or 'application/octet-stream',
                        'Metadata': {
                            'original-name': file.filename,
                            'upload-time': datetime.utcnow().isoformat(),
                            'uploaded-by': 'image-recognition-system'
                        }
                    }
                )
                
                logger.info(f"Successfully uploaded to S3: {unique_filename}")
                
                # Store in database if available (without Rekognition results)
                image_id = None
                if DATABASE_AVAILABLE:
                    try:
                        image_id = db_manager.create_image_record(
                            s3_key=unique_filename,
                            original_name=file.filename,
                            file_size=file_size
                        )
                        db_manager.log_processing_event(
                            image_id=image_id,
                            process_type='upload',
                            status='completed',
                            message=f'Uploaded to S3: {unique_filename}'
                        )
                        
                        # Set processing status to 'pending' - Lambda will update this
                        db_manager.update_processing_status(
                            image_id=image_id,
                            status='pending',
                            processed_at=None
                        )
                        
                        logger.info(f"Created database record with ID: {image_id}, processing status: pending")
                    except Exception as db_error:
                        logger.error(f"Database error: {db_error}")
                        # Continue without database - don't fail the upload
                
                # Return immediate response (no Rekognition processing here)
                uploaded_files.append({
                    'fileName': unique_filename,
                    'originalName': file.filename,
                    's3Key': unique_filename,
                    'bucket': S3_BUCKET,
                    'status': 'uploaded',
                    'processing_status': 'pending',
                    'message': 'Image uploaded successfully. Processing will complete shortly.',
                    'uploadTime': datetime.utcnow().isoformat(),
                    'imageId': image_id,
                    'fileSize': file_size,
                    'rekognition': {
                        'status': 'processing',
                        'message': 'AI analysis in progress via Lambda...',
                        'labels': [],
                        'boundingBoxes': [],
                        'faceBoxes': []
                    }
                })
                
                logger.info(f"Successfully uploaded: {file.filename} - Lambda will process asynchronously")
                
            except ClientError as e:
                logger.error(f"S3 upload failed for {file.filename}: {e}")
                uploaded_files.append({
                    'fileName': file.filename,
                    'status': 'failed',
                    'error': str(e)
                })
            except Exception as e:
                logger.error(f"Processing failed for {file.filename}: {e}")
                uploaded_files.append({
                    'fileName': file.filename,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'bucket': S3_BUCKET,
            'database_enabled': DATABASE_AVAILABLE,
            'processing_mode': 'async_lambda',
            'message': 'Images uploaded successfully. AI processing will complete in the background via Lambda.'
        })
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/images')
def get_images():
    """Get all images - results from database (processed by Lambda)"""
    try:
        logger.info("=== /api/images endpoint called ===")
        logger.info(f"DATABASE_AVAILABLE: {DATABASE_AVAILABLE}")
        logger.info(f"S3_BUCKET: {S3_BUCKET}")
        
        if DATABASE_AVAILABLE:
            logger.info("Attempting to fetch images from database...")
            images = []
            
            try:
                logger.info("Testing database connection...")
                db_manager.test_connection()
                logger.info("Database connection test successful")
                
                # Get all images from database
                logger.info("Querying database for images...")
                db_images = db_manager.get_all_images_with_detections()  # Fixed method name
                logger.info(f"Database returned {len(db_images)} images")
                
                if len(db_images) == 0:
                    logger.warning("Database returned 0 images - may be empty or query issue")
                
                for i, db_image in enumerate(db_images):
                    try:
                        logger.info(f"Processing database image {i+1}: {db_image.get('s3_key', 'unknown')}")
                        logger.info(f"Image processing status: {db_image.get('processing_status', 'unknown')}")
                        logger.info(f"Image has labels: {len(db_image.get('labels', []))}")
                        logger.info(f"Image has person detections: {len(db_image.get('person_detections', []))}")
                        logger.info(f"Image has face detections: {len(db_image.get('face_detections', []))}")
                        
                        # Generate presigned URL for image access
                        image_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET, 'Key': db_image['s3_key']},
                            ExpiresIn=3600
                        )
                        
                        # Convert database format to API format
                        rekognition_data = {
                            'status': db_image.get('processing_status', 'unknown'),
                            'labels': [
                                {'Name': label['label_name'], 'Confidence': float(label['confidence'])}
                                for label in db_image.get('labels', [])
                            ],
                            'boundingBoxes': [
                                {
                                    'Left': float(person['bbox_left']),
                                    'Top': float(person['bbox_top']),
                                    'Width': float(person['bbox_width']),
                                    'Height': float(person['bbox_height']),
                                    'confidence': float(person['confidence'])
                                }
                                for person in db_image.get('person_detections', [])
                            ],
                            'faceBoxes': []
                        }
                        
                        # Process face detections
                        for face in db_image.get('face_detections', []):
                            face_data = {
                                'Left': float(face['bbox_left']),
                                'Top': float(face['bbox_top']),
                                'Width': float(face['bbox_width']),
                                'Height': float(face['bbox_height']),
                                'confidence': float(face['confidence'])
                            }
                            
                            # Add age range if available
                            if face.get('age_low') is not None and face.get('age_high') is not None:
                                face_data['ageRange'] = {
                                    'Low': int(face['age_low']),
                                    'High': int(face['age_high'])
                                }
                            
                            # Add gender if available
                            if face.get('gender'):
                                face_data['gender'] = {
                                    'Value': face['gender'],
                                    'Confidence': float(face.get('gender_confidence', 0))
                                }
                            
                            # Add emotions
                            emotions = []
                            if face.get('primary_emotion'):
                                emotions.append({
                                    'Type': face['primary_emotion'],
                                    'Confidence': float(face.get('emotion_confidence', 0))
                                })
                            
                            if emotions:
                                face_data['emotions'] = emotions
                            
                            rekognition_data['faceBoxes'].append(face_data)
                        
                        logger.info(f"Processed recognition data - Labels: {len(rekognition_data['labels'])}, Persons: {len(rekognition_data['boundingBoxes'])}, Faces: {len(rekognition_data['faceBoxes'])}")
                        
                        image_info = {
                            'fileName': db_image['s3_key'],
                            'originalName': db_image.get('original_name', db_image['s3_key']),
                            'uploadTime': db_image['upload_time'].isoformat() if db_image.get('upload_time') else None,
                            'size': db_image.get('file_size'),
                            'url': image_url,
                            'rekognition': rekognition_data,
                            'processing_status': db_image.get('processing_status', 'unknown'),
                            'processed_at': db_image['processed_at'].isoformat() if db_image.get('processed_at') else None,
                            'imageId': db_image['id']
                        }
                        
                        images.append(image_info)
                        logger.info(f"Successfully processed image {i+1}")
                        
                    except Exception as e:
                        logger.error(f"Error processing database image {db_image.get('s3_key', 'unknown')}: {e}")
                        continue
                
                logger.info(f"Returning {len(images)} images from database")
                return jsonify({
                    'success': True,
                    'images': images,
                    'source': 'database',
                    'processing_mode': 'lambda_async',
                    'count': len(images)
                })
                
            except Exception as db_error:
                logger.error(f"Database query failed with error: {db_error}")
                logger.error(f"Database error type: {type(db_error)}")
                logger.error(f"Database available: {DATABASE_AVAILABLE}")
                logger.info("Falling back to S3 listing...")
        
        # Fallback to S3 listing (without Rekognition results)
        logger.info("Using S3 fallback for image listing")
        
        try:
            logger.info(f"Listing S3 objects in bucket: {S3_BUCKET}")
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix='uploads/'
            )
            
            logger.info(f"S3 response: {len(response.get('Contents', []))} objects found")
            
            images = []
            for i, obj in enumerate(response.get('Contents', [])):
                try:
                    logger.info(f"Processing S3 object {i+1}: {obj['Key']}")
                    
                    # Generate presigned URL
                    image_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_BUCKET, 'Key': obj['Key']},
                        ExpiresIn=3600
                    )
                    
                    image_info = {
                        'fileName': obj['Key'],
                        'originalName': obj['Key'].split('/')[-1],
                        'uploadTime': obj['LastModified'].isoformat(),
                        'size': obj['Size'],
                        'url': image_url,
                        'rekognition': {
                            'status': 'processing' if DATABASE_AVAILABLE else 'unknown',
                            'message': 'Processing by Lambda...' if DATABASE_AVAILABLE else 'Database not available - processing status unknown',
                            'labels': [],
                            'boundingBoxes': [],
                            'faceBoxes': []
                        },
                        'processing_status': 'processing' if DATABASE_AVAILABLE else 'unknown'
                    }
                    
                    images.append(image_info)
                    logger.info(f"Successfully processed S3 object {i+1}")
                    
                except Exception as e:
                    logger.error(f"Error processing S3 object {obj.get('Key', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Returning {len(images)} images from S3 fallback")
            return jsonify({
                'success': True,
                'images': images,
                'source': 's3_fallback',
                'processing_mode': 'lambda_async',
                'count': len(images),
                'message': 'Using S3 fallback - database unavailable' if not DATABASE_AVAILABLE else 'Database query failed, using S3 fallback'
            })
            
        except ClientError as e:
            logger.error(f"S3 listing failed: {e}")
            error_response = {
                'success': False,
                'error': f'S3 Error: {str(e)}',
                'images': [],
                'source': 's3_error'
            }
            logger.error(f"Returning error response: {error_response}")
            return jsonify(error_response), 500
        
    except Exception as e:
        logger.error(f"Critical error in get_images: {e}")
        error_response = {
            'success': False,
            'error': str(e),
            'images': [],
            'source': 'critical_error'
        }
        logger.error(f"Returning critical error response: {error_response}")
        return jsonify(error_response), 500

@app.route('/api/image/<path:s3_key>')
def get_image_url(s3_key):
    """Get presigned URL for a specific image"""
    try:
        logger.info(f"Getting presigned URL for: {s3_key}")
        
        # Generate presigned URL
        image_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600
        )
        
        return jsonify({
            'success': True,
            'url': image_url,
            's3_key': s3_key,
            'bucket': S3_BUCKET
        })
        
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
        return jsonify({
            'success': False,
            'error': f'S3 Error: {str(e)}',
            's3_key': s3_key
        }), 404
    except Exception as e:
        logger.error(f"Error generating presigned URL for {s3_key}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            's3_key': s3_key
        }), 500

@app.route('/api/processing-status/<image_id>')
def get_processing_status(image_id):
    """Get processing status for a specific image"""
    try:
        if not DATABASE_AVAILABLE:
            return jsonify({'error': 'Database not available'}), 503
        
        status_info = db_manager.get_processing_status(image_id)
        
        if not status_info:
            return jsonify({'error': 'Image not found'}), 404
        
        return jsonify({
            'success': True,
            'image_id': image_id,
            'processing_status': status_info['processing_status'],
            'processed_at': status_info['processed_at'].isoformat() if status_info['processed_at'] else None,
            'upload_time': status_info['upload_time'].isoformat() if status_info['upload_time'] else None,
            'has_results': status_info['processing_status'] == 'completed'
        })
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/processing-status/batch', methods=['POST'])
def get_batch_processing_status():
    """Get processing status for multiple images"""
    try:
        if not DATABASE_AVAILABLE:
            return jsonify({'error': 'Database not available'}), 503
        
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        
        if not image_ids:
            return jsonify({'error': 'No image IDs provided'}), 400
        
        statuses = {}
        for image_id in image_ids:
            status_info = db_manager.get_processing_status(image_id)
            if status_info:
                statuses[str(image_id)] = {
                    'processing_status': status_info['processing_status'],
                    'processed_at': status_info['processed_at'].isoformat() if status_info['processed_at'] else None,
                    'has_results': status_info['processing_status'] == 'completed'
                }
        
        return jsonify({
            'success': True,
            'statuses': statuses
        })
        
    except Exception as e:
        logger.error(f"Error getting batch processing status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'processing_mode': 'lambda_async',
            'components': {}
        }
        
        # Check S3 connectivity
        try:
            logger.info("Checking S3 connectivity")
            s3_client.head_bucket(Bucket=S3_BUCKET)
            status['components']['s3'] = {
                'status': 'healthy',
                'bucket': S3_BUCKET,
                'message': 'S3 bucket accessible'
            }
            logger.info("S3 check: healthy")
        except ClientError as e:
            logger.error(f"S3 check failed: {e}")
            status['components']['s3'] = {
                'status': 'unhealthy',
                'bucket': S3_BUCKET,
                'message': f'S3 Error: {str(e)}',
                'error_code': e.response['Error']['Code']
            }
            status['status'] = 'degraded'
        except Exception as e:
            logger.error(f"S3 connection error: {e}")
            status['components']['s3'] = {
                'status': 'unhealthy',
                'bucket': S3_BUCKET,
                'message': f'S3 Connection Error: {str(e)}'
            }
            status['status'] = 'degraded'
        
        # Check database connectivity
        if DATABASE_AVAILABLE:
            try:
                logger.info("Checking database connectivity")
                db_manager.test_connection()
                status['components']['database'] = {
                    'status': 'healthy',
                    'message': 'Database connection successful'
                }
                logger.info("Database check: healthy")
            except Exception as e:
                logger.error(f"Database check failed: {e}")
                status['components']['database'] = {
                    'status': 'unhealthy',
                    'message': f'Database Error: {str(e)}'
                }
                status['status'] = 'degraded'
        else:
            status['components']['database'] = {
                'status': 'unavailable',
                'message': 'Database module not loaded'
            }
        
        # Note: Rekognition is now handled by Lambda, not checked here
        status['components']['rekognition'] = {
            'status': 'lambda_managed',
            'message': 'Image processing handled by Lambda function'
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/status/infrastructure')
def infrastructure_status():
    """Infrastructure status endpoint (expected by frontend)"""
    try:
        # Get basic health status
        health_response = health_check()
        health_data = health_response.get_json()
        
        # Transform to infrastructure status format
        infrastructure_status = {
            'overall': health_data.get('status', 'unknown'),
            'timestamp': health_data.get('timestamp'),
            'processing_mode': health_data.get('processing_mode', 'lambda_async'),
            'services': {
                's3': health_data.get('components', {}).get('s3', {'status': 'unknown'}),
                'database': health_data.get('components', {}).get('database', {'status': 'unknown'}),
                'lambda': health_data.get('components', {}).get('rekognition', {'status': 'lambda_managed'})
            }
        }
        
        return jsonify(infrastructure_status)
        
    except Exception as e:
        logger.error(f"Infrastructure status error: {e}")
        return jsonify({
            'overall': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/config')
def get_config():
    """Get application configuration"""
    return jsonify({
        'bucket': S3_BUCKET,
        'region': config.get('region', 'us-west-2'),
        'database_enabled': DATABASE_AVAILABLE,
        'processing_mode': 'lambda_async',
        'features': {
            'async_processing': True,
            'lambda_rekognition': True,
            'database_storage': DATABASE_AVAILABLE,
            'real_time_status': DATABASE_AVAILABLE
        }
    })

@app.route('/favicon.ico')
def favicon():
    """Return empty favicon to prevent 502 errors"""
    return '', 204

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

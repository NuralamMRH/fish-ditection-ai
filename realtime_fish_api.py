#!/usr/bin/env python3
"""
Real-time Fish Detection API with Multi-Model Support & Callback System
======================================================================

Enhanced API endpoints:
- POST /api/detect - Upload image for fish detection and analysis
- GET /api/status - Get real-time detection status
- POST /api/callback - Set callback URL for data submission
- GET /api/results/{session_id} - Get detection results for session
- POST /api/submit/{session_id} - Submit results to callback URL
- GET /api/health - API health check
- POST /api/camera/start - Start camera feed
- POST /api/camera/stop - Stop camera feed
- GET /api/camera/feed - Get camera feed status
"""

import os
import sys
import cv2
import json
import uuid
import time
import requests
import threading
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import math

# Add detector paths
sys.path.insert(0, './detector_v12')
sys.path.insert(0, './detector_v10_m3')
sys.path.insert(0, './classification_rectangle_v7-1')

try:
    from local_inference import LocalYOLOv12Fish
except ImportError:
    print("Warning: Could not import LocalYOLOv12Fish")
    LocalYOLOv12Fish = None

try:
    from inference import YOLOInference
except ImportError:
    print("Warning: Could not import YOLOInference (YOLO10)")
    YOLOInference = None

try:
    from inference import EmbeddingClassifier
except ImportError:
    print("Warning: Could not import EmbeddingClassifier")
    EmbeddingClassifier = None

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Callback-URL"]
    }
})

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global storage for sessions and callbacks
active_sessions = {}
callback_urls = {}
camera_status = {
    'active': False,
    'current_frame': None,
    'last_detection': None,
    'detection_count': 0
}

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
        return float(obj)
    elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

class EnhancedFishDetectionAPI:
    """Enhanced Fish Detection API with multi-model support and 3D measurement."""
    
    def __init__(self):
        """Initialize the enhanced fish detection API."""
        self.detectors = []
        self.classifier = None
        self.measurement_system = AdvancedMeasurementSystem()
        
        # Initialize YOLO12 detector (primary)
        if LocalYOLOv12Fish:
            try:
                detector_yolo12 = LocalYOLOv12Fish(confidence=0.1)
                self.detectors.append(('YOLO12', detector_yolo12))
                print("✅ YOLO12 fish detector loaded")
            except Exception as e:
                print(f"❌ Failed to load YOLO12 detector: {e}")
        
        # Initialize YOLO10 detector (fallback)
        if YOLOInference:
            try:
                detector_yolo10 = YOLOInference(
                    model_path='./detector_v10_m3/model.ts',
                    conf_threshold=0.05,
                    nms_threshold=0.3,
                    yolo_ver='v10'
                )
                self.detectors.append(('YOLO10', detector_yolo10))
                print("✅ YOLO10 fish detector loaded")
            except Exception as e:
                print(f"❌ Failed to load YOLO10 detector: {e}")
        
        # Initialize classifier
        if EmbeddingClassifier:
            try:
                self.classifier = EmbeddingClassifier(
                    model_path='./classification_rectangle_v7-1/model.ts',
                    data_set_path='./classification_rectangle_v7-1/database.pt',
                    device='cpu'
                )
                print("✅ Fish classifier loaded")
            except Exception as e:
                print(f"❌ Failed to load classifier: {e}")
    
    def detect_fish_multi_model(self, image):
        """Detect fish using multiple models for better reliability."""
        all_detections = []
        
        for model_name, detector in self.detectors:
            try:
                print(f"🔍 Trying {model_name} detection...")
                
                if model_name == 'YOLO12':
                    detections = detector.predict(image)
                    fish_list = detections[0] if detections and detections[0] else []
                    
                    if fish_list:
                        print(f"✅ {model_name} detected {len(fish_list)} fish")
                        for fish in fish_list:
                            bbox = fish.get_box()
                            confidence = fish.get_score()
                            species = fish.get_class_name()
                            
                            detection_data = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'species': species,
                                'model': model_name
                            }
                            all_detections.append(detection_data)
                        break
                
                elif model_name == 'YOLO10':
                    detections = detector.predict(image)
                    fish_list = detections[0] if detections and detections[0] else []
                    
                    if fish_list:
                        print(f"✅ {model_name} detected {len(fish_list)} fish")
                        for fish in fish_list:
                            bbox = fish.get_box()
                            confidence = fish.get_score()
                            species = 'Fish'
                            
                            detection_data = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'species': species,
                                'model': model_name
                            }
                            all_detections.append(detection_data)
                        break
                        
            except Exception as e:
                print(f"❌ {model_name} detection error: {e}")
                continue
        
        return all_detections
    
    def classify_species(self, image, fish_bbox):
        """Classify fish species if classifier available."""
        if not self.classifier:
            return None
        
        try:
            x1, y1, x2, y2 = map(int, fish_bbox)
            fish_crop = image[y1:y2, x1:x2]
            if fish_crop.size > 0:
                results = self.classifier.inference_numpy(fish_crop)
                if results and len(results) > 0:
                    return {
                        'predicted_species': results[0]['name'],
                        'confidence': results[0]['accuracy'],
                        'species_id': results[0]['species_id']
                    }
                return None
        except Exception as e:
            print(f"Classification error: {e}")
            return None
    
    def analyze_image(self, image):
        """Comprehensive image analysis with multi-model detection."""
        try:
            self.measurement_system.auto_calibrate_camera(image)
            
            detected_fish_data = self.detect_fish_multi_model(image)
            
            if detected_fish_data:
                processed_fish = []
                
                for i, fish_data in enumerate(detected_fish_data):
                    bbox = fish_data['bbox']
                    confidence = fish_data['confidence']
                    species = fish_data['species']
                    model_used = fish_data['model']
                    
                    # Calculate measurements
                    distance_cm = self.measurement_system.estimate_depth_monocular(bbox, image.shape)
                    dimensions = self.measurement_system.calculate_3d_dimensions(bbox, distance_cm, species)
                    weight_info = self.measurement_system.calculate_advanced_weight(dimensions, species)
                    
                    # Species classification
                    species_classification = self.classify_species(image, bbox)
                    
                    final_species = species
                    if species_classification and species_classification.get('predicted_species'):
                        final_species = species_classification['predicted_species']
                    
                    fish_complete = {
                        'id': i,
                        'species': final_species,
                        'confidence': float(confidence),
                        'bbox': [float(x) for x in bbox],
                        'distance_cm': round(float(distance_cm), 1),
                        'dimensions': convert_numpy_types(dimensions),
                        'weight': convert_numpy_types(weight_info),
                        'species_classification': convert_numpy_types(species_classification),
                        'timestamp': datetime.now().isoformat(),
                        'detected_by': model_used
                    }
                    
                    processed_fish.append(fish_complete)
                
                return {
                    'success': True,
                    'fish_count': len(processed_fish),
                    'fish': processed_fish,
                    'timestamp': datetime.now().isoformat(),
                    'models_used': list(set([f['detected_by'] for f in processed_fish]))
                }
            else:
                return {
                    'success': False,
                    'fish_count': 0,
                    'fish': [],
                    'message': 'No fish detected by any model'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fish_count': 0,
                'fish': []
            }

class AdvancedMeasurementSystem:
    """Advanced 3D measurement system for fish detection."""
    
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.focal_length_mm = 4.0
        self.sensor_size_mm = (5.76, 4.29)
        self.calibrated = False
        self.pixels_per_cm = 50
        
        # Species weight formulas
        self.weight_formulas = {
            'Tuna': lambda l, h, g: (l * h * g * 0.00028),
            'Bass': lambda l, h, g: (l * h * g * 0.00025),
            'Salmon': lambda l, h, g: (l * h * g * 0.00026),
            'Catfish': lambda l, h, g: (l * h * g * 0.00030),
            'Snapper': lambda l, h, g: (l * h * g * 0.00027),
            'Mackerel': lambda l, h, g: (l * h * g * 0.00023),
            'Grouper': lambda l, h, g: (l * h * g * 0.00032),
            'default': lambda l, h, g: (l * h * g * 0.00027)
        }
        
        self.girth_ratios = {
            'Tuna': 0.65, 'Bass': 0.45, 'Salmon': 0.40, 'Catfish': 0.55,
            'Snapper': 0.42, 'Mackerel': 0.35, 'Grouper': 0.50, 'default': 0.45
        }
    
    def auto_calibrate_camera(self, frame):
        """Auto-calibrate camera using frame dimensions."""
        try:
            h, w = frame.shape[:2]
            
            if not self.calibrated:
                diagonal_pixels = math.sqrt(w*w + h*h)
                diagonal_inches = 6.0
                diagonal_cm = diagonal_inches * 2.54
                self.pixels_per_cm = diagonal_pixels / diagonal_cm
                
                fx = fy = w * 0.8
                cx, cy = w/2, h/2
                
                self.camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                self.dist_coeffs = np.zeros((4,1))
                self.calibrated = True
                
        except Exception as e:
            print(f"Calibration error: {e}")
    
    def estimate_depth_monocular(self, fish_bbox, frame_shape):
        """Estimate depth using monocular cues."""
        try:
            x1, y1, x2, y2 = fish_bbox
            fish_width_pixels = x2 - x1
            fish_height_pixels = y2 - y1
            frame_height, frame_width = frame_shape[:2]
            
            fish_area = fish_width_pixels * fish_height_pixels
            frame_area = frame_width * frame_height
            area_ratio = fish_area / frame_area
            
            depth_size = 100 / math.sqrt(area_ratio)
            
            fish_center_y = (y1 + y2) / 2
            vertical_position = fish_center_y / frame_height
            depth_position = 50 + (vertical_position * 100)
            
            estimated_depth = (depth_size + depth_position) / 2
            return max(20, min(300, estimated_depth))
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return 80
    
    def calculate_3d_dimensions(self, fish_bbox, depth_cm, species='default'):
        """Calculate accurate 3D dimensions of fish."""
        try:
            x1, y1, x2, y2 = fish_bbox
            
            fish_width_pixels = x2 - x1
            fish_height_pixels = y2 - y1
            
            if self.camera_matrix is not None:
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                
                width_cm = (fish_width_pixels * depth_cm) / fx
                height_cm = (fish_height_pixels * depth_cm) / fy
            else:
                width_cm = fish_width_pixels / self.pixels_per_cm
                height_cm = fish_height_pixels / self.pixels_per_cm
            
            length_cm = max(width_cm, height_cm)
            body_height_cm = min(width_cm, height_cm)
            
            girth_ratio = self.girth_ratios.get(species, self.girth_ratios['default'])
            girth_cm = length_cm * girth_ratio
            
            volume_cm3 = (4/3) * math.pi * (length_cm/2) * (body_height_cm/2) * (girth_cm/4)
            
            return {
                'total_length_cm': length_cm,
                'fork_length_cm': length_cm * 0.8,
                'standard_length_cm': length_cm * 0.59,
                'body_height_cm': body_height_cm,
                'estimated_girth_cm': girth_cm,
                'volume_cm3': volume_cm3,
                'length_inches': length_cm / 2.54,
                'height_inches': body_height_cm / 2.54,
                'girth_inches': girth_cm / 2.54
            }
            
        except Exception as e:
            print(f"3D calculation error: {e}")
            return None
    
    def calculate_advanced_weight(self, dimensions, species='default'):
        """Calculate fish weight using advanced formulas."""
        try:
            if not dimensions:
                return None
            
            length_cm = dimensions['total_length_cm']
            height_cm = dimensions['body_height_cm']
            girth_cm = dimensions['estimated_girth_cm']
            
            weight_formula = self.weight_formulas.get(species, self.weight_formulas['default'])
            weight_kg = weight_formula(length_cm, height_cm, girth_cm)
            
            standard_weight_kg = 0.00001 * (length_cm ** 3.1)
            volume_weight_kg = dimensions['volume_cm3'] * 0.001
            
            final_weight_kg = (weight_kg + standard_weight_kg + volume_weight_kg) / 3
            
            return {
                'weight_kg': final_weight_kg,
                'weight_pounds': final_weight_kg * 2.20462,
                'weight_grams': final_weight_kg * 1000,
                'weight_ounces': final_weight_kg * 35.274,
                'species_weight_kg': weight_kg,
                'standard_weight_kg': standard_weight_kg,
                'volume_weight_kg': volume_weight_kg,
                'calculation_methods': 3
            }
            
        except Exception as e:
            print(f"Weight calculation error: {e}")
            return None

# Initialize the API detector
api_detector = EnhancedFishDetectionAPI()

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API ENDPOINTS

@app.route('/api/detect', methods=['POST'])
def detect_fish():
    """Upload image for fish detection and analysis."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        # Get callback URL from multiple sources (query params, headers, form data)
        callback_url = (
            request.args.get('callback_url') or 
            request.headers.get('X-Callback-URL') or 
            request.form.get('callback_url')
        )
        
        print(f"🔗 Callback URL from request: {callback_url}")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(f"{session_id}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read and process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not read image file'}), 400
        
        # Analyze image
        result = api_detector.analyze_image(image)
        
        # Store session data
        session_data = {
            'session_id': session_id,
            'filename': filename,
            'filepath': filepath,
            'result': result,
            'created_at': datetime.now().isoformat(),
            'callback_url': callback_url
        }
        
        active_sessions[session_id] = session_data
        if callback_url:
            callback_urls[session_id] = callback_url
            print(f"✅ Callback URL set for session {session_id}: {callback_url}")
        
        # Save results
        result_file = os.path.join(RESULTS_FOLDER, f"{session_id}_result.json")
        with open(result_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        # Return result with session info
        response_data = {
            'session_id': session_id,
            'success': result['success'],
            'fish_count': result['fish_count'],
            'fish': result['fish'],
            'callback_url': callback_url,
            'submit_url': f'/api/submit/{session_id}' if callback_url else None,
            'timestamp': result.get('timestamp'),
            'models_used': result.get('models_used', [])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get real-time detection status."""
    return jsonify({
        'active_sessions': len(active_sessions),
        'camera_active': camera_status['active'],
        'detection_count': camera_status['detection_count'],
        'last_detection': camera_status['last_detection'],
        'models_available': len(api_detector.detectors),
        'classifier_available': api_detector.classifier is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/callback', methods=['POST'])
def set_callback():
    """Set callback URL for a session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        callback_url = data.get('callback_url')
        
        if not session_id or not callback_url:
            return jsonify({'success': False, 'error': 'Missing session_id or callback_url'}), 400
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        callback_urls[session_id] = callback_url
        active_sessions[session_id]['callback_url'] = callback_url
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'callback_url': callback_url,
            'submit_url': f'/api/submit/{session_id}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/results/<session_id>', methods=['GET'])
def get_results(session_id):
    """Get detection results for a session."""
    try:
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session_data = active_sessions[session_id]
        return jsonify({
            'session_id': session_id,
            'result': session_data['result'],
            'created_at': session_data['created_at'],
            'callback_url': session_data.get('callback_url'),
            'submit_url': f'/api/submit/{session_id}' if session_data.get('callback_url') else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/submit/<session_id>', methods=['POST'])
def submit_results(session_id):
    """Submit results to callback URL via REST API and redirect."""
    try:
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session_data = active_sessions[session_id]
        callback_url = session_data.get('callback_url')
        
        if not callback_url:
            return jsonify({'success': False, 'error': 'No callback URL set for this session'}), 400
        
        # Get additional data from request
        request_data = request.get_json() if request.is_json else {}
        
        # Prepare comprehensive data for callback REST API
        callback_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'fish_data': session_data['result'],
            'source': 'fish_detection_api',
            'version': '2.0',
            'api_endpoint': f"{request.host_url}api/results/{session_id}",
            'total_fish_count': session_data['result']['fish_count'],
            'detection_success': session_data['result']['success']
        }
        
        # Add fish summary for quick access
        if session_data['result']['fish']:
            fish_summary = []
            total_weight_kg = 0
            
            for fish in session_data['result']['fish']:
                fish_info = {
                    'species': fish['species'],
                    'confidence': fish['confidence'],
                    'weight_kg': fish.get('weight', {}).get('weight_kg', 0),
                    'weight_pounds': fish.get('weight', {}).get('weight_pounds', 0),
                    'length_cm': fish.get('dimensions', {}).get('total_length_cm', 0),
                    'length_inches': fish.get('dimensions', {}).get('length_inches', 0),
                    'detected_by': fish.get('detected_by', 'Unknown')
                }
                fish_summary.append(fish_info)
                total_weight_kg += fish_info['weight_kg']
            
            callback_data.update({
                'fish_summary': fish_summary,
                'total_weight_kg': total_weight_kg,
                'total_weight_pounds': total_weight_kg * 2.20462
            })
        
        # Add any additional user data
        if request_data:
            callback_data['user_data'] = request_data
        
        print(f"📤 Submitting to callback URL: {callback_url}")
        print(f"📊 Data: {json.dumps(callback_data, indent=2, default=str)[:500]}...")
        
        # Submit to callback URL via REST API
        callback_success = False
        callback_response = None
        callback_error = None
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Fish-Detection-API/2.0',
                'X-API-Source': 'fish-detection-api'
            }
            
            response = requests.post(
                callback_url, 
                json=callback_data, 
                headers=headers,
                timeout=30
            )
            
            callback_success = response.status_code in [200, 201, 202]
            callback_response = {
                'status_code': response.status_code,
                'response_text': response.text[:1000] if response.text else None,
                'headers': dict(response.headers)
            }
            
            if callback_success:
                print(f"✅ Callback submission successful: {response.status_code}")
            else:
                print(f"❌ Callback submission failed: {response.status_code} - {response.text}")
            
        except requests.exceptions.Timeout:
            callback_error = "Callback request timed out after 30 seconds"
            print(f"⏱️ {callback_error}")
        except requests.exceptions.ConnectionError:
            callback_error = "Could not connect to callback URL"
            print(f"🔌 {callback_error}")
        except Exception as e:
            callback_error = f"Callback request failed: {str(e)}"
            print(f"❌ {callback_error}")
        
        # Prepare response
        response_data = {
            'success': True,
            'session_id': session_id,
            'callback_url': callback_url,
            'callback_submitted': callback_success,
            'callback_response': callback_response,
            'callback_error': callback_error,
            'fish_count': session_data['result']['fish_count'],
            'redirect_url': callback_url
        }
        
        # Handle different response types
        if request.headers.get('Content-Type') == 'application/json' or request.args.get('format') == 'json':
            # API call - return JSON
            return jsonify(response_data)
        else:
            # Browser call - redirect to callback URL
            if callback_success:
                print(f"🔄 Redirecting to: {callback_url}")
                return redirect(callback_url)
            else:
                # If callback failed, return error page
                error_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Callback Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                        .container {{ background: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: 0 auto; }}
                        .error {{ color: #d32f2f; }}
                        .info {{ color: #1976d2; }}
                        .btn {{ display: inline-block; padding: 10px 20px; background: #1976d2; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🚨 Callback Submission Failed</h1>
                        <p class="error">Failed to submit fish data to: <strong>{callback_url}</strong></p>
                        <p><strong>Error:</strong> {callback_error or 'Unknown error'}</p>
                        <p class="info">Your fish detection was successful with {session_data['result']['fish_count']} fish detected.</p>
                        <a href="/" class="btn">🏠 Return to API</a>
                        <a href="/demo" class="btn">🎮 Try Demo</a>
                        <a href="/api/results/{session_id}" class="btn">📊 View Results</a>
                    </div>
                </body>
                </html>
                """
                return error_html, 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera feed for real-time detection."""
    try:
        camera_status['active'] = True
        camera_status['detection_count'] = 0
        camera_status['last_detection'] = None
        
        return jsonify({
            'success': True,
            'message': 'Camera started',
            'status': camera_status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera feed."""
    try:
        camera_status['active'] = False
        camera_status['current_frame'] = None
        
        return jsonify({
            'success': True,
            'message': 'Camera stopped',
            'status': camera_status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera/feed', methods=['GET'])
def get_camera_feed():
    """Get camera feed status and latest detection."""
    return jsonify({
        'active': camera_status['active'],
        'detection_count': camera_status['detection_count'],
        'last_detection': camera_status['last_detection'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'detectors': len(api_detector.detectors),
            'classifier': api_detector.classifier is not None
        },
        'sessions': {
            'active': len(active_sessions),
            'callbacks': len(callback_urls)
        },
        'camera': camera_status['active']
    })

@app.route('/api/models', methods=['GET'])
def model_info():
    """Get model information."""
    detector_info = []
    for model_name, detector in api_detector.detectors:
        detector_info.append({
            'name': model_name,
            'type': 'Fish Detector',
            'status': 'loaded'
        })
    
    return jsonify({
        'detectors': detector_info,
        'classifier': {
            'available': api_detector.classifier is not None,
            'type': 'Fish Species Classifier'
        },
        'measurement_system': {
            'available': True,
            'features': ['3D measurement', 'weight calculation', 'depth estimation']
        }
    })

@app.route('/demo', methods=['GET'])
def demo_interface():
    """Demo interface for testing the API with callback functionality."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐟 Real-time Fish Detection API Demo</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #ddd; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
            .upload-area.dragover { border-color: #007bff; background: #f8f9ff; }
            .results { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
            .fish-item { margin: 10px 0; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #007bff; }
            .btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #0056b3; }
            .btn-success { background: #28a745; }
            .btn-success:hover { background: #218838; }
            .input-group { margin: 10px 0; }
            .input-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .input-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .url-example { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐟 Real-time Fish Detection API Demo</h1>
            
            <div class="url-example">
                <h4>📝 How to use Callback URL via Query Parameter:</h4>
                <p><strong>Example:</strong> <code>http://localhost:5009/demo?callback_url=https://www.itrucksea.com/fishing-log/batch</code></p>
                <p>The API will automatically detect and use the callback URL from the query parameter.</p>
            </div>
            
            <div class="input-group">
                <label for="callbackUrl">Callback URL:</label>
                <input type="url" id="callbackUrl" placeholder="https://www.itrucksea.com/fishing-log/batch" 
                       value="">
            </div>
            
            <div class="upload-area" id="uploadArea">
                <p>Drag & drop fish image here or click to select</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <button class="btn" onclick="document.getElementById('fileInput').click()">Select Image</button>
            </div>
            
            <div id="status"></div>
            <div id="results"></div>
        </div>
        
        <script>
            let currentSessionId = null;
            
            // Check for callback URL in query parameters
            const urlParams = new URLSearchParams(window.location.search);
            const callbackFromQuery = urlParams.get('callback_url');
            if (callbackFromQuery) {
                document.getElementById('callbackUrl').value = callbackFromQuery;
                showStatus(`Callback URL loaded from query: ${callbackFromQuery}`, 'success');
            }
            
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const statusDiv = document.getElementById('status');
            const resultsDiv = document.getElementById('results');
            
            // Drag and drop handlers
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadImage(files[0]);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    uploadImage(e.target.files[0]);
                }
            });
            
            function showStatus(message, type = 'success') {
                statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            }
            
            function uploadImage(file) {
                const formData = new FormData();
                formData.append('image', file);
                
                const callbackUrl = document.getElementById('callbackUrl').value;
                
                // Build URL with callback as query parameter
                let uploadUrl = '/api/detect';
                if (callbackUrl) {
                    uploadUrl += `?callback_url=${encodeURIComponent(callbackUrl)}`;
                }
                
                showStatus('Analyzing image...', 'success');
                resultsDiv.innerHTML = '';
                
                fetch(uploadUrl, {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    currentSessionId = data.session_id;
                    
                    if (data.success) {
                        showStatus(`Analysis complete! Found ${data.fish_count} fish.`, 'success');
                        displayResults(data);
                    } else {
                        showStatus(`Analysis failed: ${data.error || 'Unknown error'}`, 'error');
                    }
                })
                .catch(error => {
                    showStatus(`Error: ${error.message}`, 'error');
                });
            }
            
            function displayResults(data) {
                let html = `
                    <div class="results">
                        <h3>🎣 Detection Results</h3>
                        <p><strong>Session ID:</strong> ${data.session_id}</p>
                        <p><strong>Fish Count:</strong> ${data.fish_count}</p>
                        <p><strong>Models Used:</strong> ${data.models_used ? data.models_used.join(', ') : 'N/A'}</p>
                `;
                
                if (data.callback_url) {
                    html += `
                        <p><strong>Callback URL:</strong> ${data.callback_url}</p>
                        <button class="btn btn-success" onclick="submitToCallback()">
                            📤 Submit to Callback URL & Redirect
                        </button>
                        <button class="btn" onclick="submitToCallbackAPI()">
                            📋 Submit to Callback (API Only)
                        </button>
                    `;
                }
                
                if (data.fish && data.fish.length > 0) {
                    html += '<h4>🐟 Fish Details:</h4>';
                    
                    data.fish.forEach((fish, index) => {
                        html += `
                            <div class="fish-item">
                                <h5>Fish ${index + 1}: ${fish.species}</h5>
                                <p><strong>Confidence:</strong> ${(fish.confidence * 100).toFixed(1)}%</p>
                                <p><strong>Weight:</strong> ${fish.weight?.weight_kg?.toFixed(3) || 'N/A'} kg 
                                   (${fish.weight?.weight_pounds?.toFixed(2) || 'N/A'} lbs)</p>
                                <p><strong>Length:</strong> ${fish.dimensions?.total_length_cm?.toFixed(1) || 'N/A'} cm 
                                   (${fish.dimensions?.length_inches?.toFixed(1) || 'N/A'} inches)</p>
                                <p><strong>Distance:</strong> ${fish.distance_cm} cm</p>
                                <p><strong>Detected by:</strong> ${fish.detected_by}</p>
                            </div>
                        `;
                    });
                }
                
                html += '</div>';
                resultsDiv.innerHTML = html;
            }
            
            function submitToCallback() {
                if (!currentSessionId) {
                    showStatus('No session to submit', 'error');
                    return;
                }
                
                showStatus('Submitting to callback URL and redirecting...', 'success');
                
                // This will redirect to callback URL
                window.location.href = `/api/submit/${currentSessionId}`;
            }
            
            function submitToCallbackAPI() {
                if (!currentSessionId) {
                    showStatus('No session to submit', 'error');
                    return;
                }
                
                showStatus('Submitting to callback URL (API only)...', 'success');
                
                fetch(`/api/submit/${currentSessionId}?format=json`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        user_info: 'Demo User',
                        source: 'demo_interface'
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const status = data.callback_submitted ? 'successful' : 'failed';
                        showStatus(`Callback submission ${status}! Check console for details.`, 
                                 data.callback_submitted ? 'success' : 'error');
                        console.log('Callback Response:', data);
                    } else {
                        showStatus(`Submission failed: ${data.error}`, 'error');
                    }
                })
                .catch(error => {
                    showStatus(`Submission error: ${error.message}`, 'error');
                });
            }
            
            // Check API status on load
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    console.log('API Status:', data);
                });
        </script>
    </body>
    </html>
    '''

@app.route('/', methods=['GET'])
def api_info():
    """API information and documentation."""
    return jsonify({
        'name': 'Real-time Fish Detection API',
        'version': '2.0',
        'description': 'Enhanced multi-model fish detection with callback system',
        'endpoints': {
            'POST /api/detect': 'Upload image for fish detection',
            'GET /api/status': 'Get real-time detection status',
            'POST /api/callback': 'Set callback URL for session',
            'GET /api/results/{session_id}': 'Get detection results',
            'POST /api/submit/{session_id}': 'Submit results to callback URL',
            'POST /api/camera/start': 'Start camera feed',
            'POST /api/camera/stop': 'Stop camera feed',
            'GET /api/camera/feed': 'Get camera feed status',
            'GET /api/health': 'API health check',
            'GET /api/models': 'Model information',
            'GET /demo': 'Demo interface'
        },
        'features': [
            'Multi-model fish detection (YOLO12 + YOLO10)',
            'Species classification',
            '3D measurement and weight calculation',
            'Real-time camera integration',
            'Callback URL system for data submission',
            'Session management',
            'Enhanced measurement system'
        ]
    })

if __name__ == '__main__':
    print("🐟 Starting Real-time Fish Detection API...")
    print("=" * 60)
    
    print("🔍 Model Status:")
    if api_detector.detectors:
        for model_name, model_instance in api_detector.detectors:
            print(f"✅ {model_name} Fish Detector: Ready")
    else:
        print("❌ No Fish Detectors Available")
    
    if api_detector.classifier:
        print("✅ Fish Classifier: Ready")
    else:
        print("❌ Fish Classifier: Not available")
    
    print("✅ 3D Measurement System: Ready")
    print("✅ Callback System: Ready")
    print("✅ Session Management: Ready")
    print("✅ Real-time API: Ready")
    
    print(f"\n🚀 API Server starting...")
    print(f"📡 API Base URL: http://localhost:5009")
    print(f"🎮 Demo Interface: http://localhost:5009/demo")
    print(f"📚 API Documentation: http://localhost:5009/")
    print(f"🔍 Health Check: http://localhost:5009/api/health")
    print(f"⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5009, debug=True, threaded=True) 
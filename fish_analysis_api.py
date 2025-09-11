#!/usr/bin/env python3
"""
Comprehensive Fish Analysis API
==============================

POST /analyze - Upload image and get complete fish analysis
GET /health - API health check
GET /models - Model information
"""

import os
import sys
import cv2
import json
import uuid
import time
import subprocess
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add detector_v12 to path
sys.path.insert(0, './detector_v12')
from local_inference import LocalYOLOv12Fish

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize YOLOv12 detector
try:
    yolo_detector = LocalYOLOv12Fish(confidence=0.3)
    print("✅ YOLOv12 Local Model loaded successfully")
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"❌ Failed to load YOLOv12: {e}")
    yolo_detector = None
    YOLO_AVAILABLE = False

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_fish_crop(crop_path):
    """Classify fish crop using existing classifier."""
    try:
        class_script = f"""
import sys, os, cv2, json
os.chdir('./classification_rectangle_v7-1')
from inference import EmbeddingClassifier
classifier = EmbeddingClassifier('./model.ts', './database.pt')
image = cv2.imread('../{crop_path}')
results = classifier.inference_numpy(image)

if results and len(results) > 0:
    result = {{
        'species': results[0]['name'],
        'accuracy': float(results[0]['accuracy']),
        'species_id': results[0]['species_id'],
        'common_name': results[0].get('common_name', results[0]['name']),
        'scientific_name': results[0].get('scientific_name', results[0]['name'])
    }}
    print('CLASS_RESULT:' + json.dumps(result))
else:
    print('CLASS_RESULT:' + json.dumps({{'species': 'Unknown', 'accuracy': 0.0}}))
"""
        
        class_result = subprocess.run([sys.executable, "-c", class_script], 
                                    capture_output=True, text=True, timeout=30)
        
        class_output = None
        for line in class_result.stdout.split('\n'):
            if line.startswith('CLASS_RESULT:'):
                class_output = json.loads(line.replace('CLASS_RESULT:', ''))
                break
        
        if not class_output:
            class_output = {
                'species': 'Unknown', 
                'accuracy': 0.0,
                'species_id': -1,
                'common_name': 'Unknown',
                'scientific_name': 'Unknown'
            }
        
        return class_output
        
    except Exception as e:
        print(f"Classification error: {e}")
        return {
            'species': 'Classification Failed', 
            'accuracy': 0.0,
            'species_id': -1,
            'common_name': 'Error',
            'scientific_name': 'Error',
            'error': str(e)
        }

def create_enhanced_polygon(crop_path, box):
    """Create enhanced polygon from fish crop."""
    try:
        x1, y1, x2, y2 = box
        
        # Default polygon (rectangle)
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        # Try to create enhanced polygon
        crop_image = cv2.imread(crop_path)
        if crop_image is not None:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                enhanced_polygon = []
                for point in simplified_contour:
                    px, py = point[0]
                    original_x = x1 + px
                    original_y = y1 + py
                    enhanced_polygon.append([int(original_x), int(original_y)])
                
                if len(enhanced_polygon) >= 4:
                    polygon = enhanced_polygon
        
        return polygon
        
    except Exception as e:
        print(f"Polygon creation error: {e}")
        x1, y1, x2, y2 = box
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def calculate_fish_metrics(box, image_shape):
    """Calculate fish size and position metrics."""
    try:
        x1, y1, x2, y2 = box
        img_height, img_width = image_shape[:2]
        
        # Fish dimensions
        fish_width = x2 - x1
        fish_height = y2 - y1
        fish_area = fish_width * fish_height
        
        # Relative to image
        relative_width = fish_width / img_width
        relative_height = fish_height / img_height
        relative_area = fish_area / (img_width * img_height)
        
        # Position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        relative_center_x = center_x / img_width
        relative_center_y = center_y / img_height
        
        # Size category
        if relative_area > 0.3:
            size_category = "Large"
        elif relative_area > 0.1:
            size_category = "Medium"
        elif relative_area > 0.05:
            size_category = "Small"
        else:
            size_category = "Very Small"
        
        return {
            "dimensions": {
                "width_pixels": int(fish_width),
                "height_pixels": int(fish_height),
                "area_pixels": int(fish_area)
            },
            "relative_size": {
                "width_ratio": round(relative_width, 4),
                "height_ratio": round(relative_height, 4),
                "area_ratio": round(relative_area, 4)
            },
            "position": {
                "center_x": int(center_x),
                "center_y": int(center_y),
                "relative_center_x": round(relative_center_x, 4),
                "relative_center_y": round(relative_center_y, 4)
            },
            "size_category": size_category
        }
        
    except Exception as e:
        return {"error": f"Metrics calculation failed: {e}"}

def create_annotated_image(image_path, analysis_result):
    """Create annotated image with all detection results."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Colors for different fish
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]
        
        for i, fish in enumerate(analysis_result.get("detections", [])):
            color = colors[i % len(colors)]
            polygon = fish.get("polygon", [])
            
            # Draw polygon
            if len(polygon) >= 4:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Semi-transparent overlay
                overlay = image.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                
                # Polygon outline
                cv2.polylines(image, [pts], True, color, 3)
            
            # Labels
            box = fish.get("detection", {}).get("bounding_box", [0, 0, 100, 100])
            x1, y1, x2, y2 = box
            
            # Fish ID
            cv2.putText(image, f"Fish #{fish.get('fish_id', i+1)}", 
                       (x1, y1-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # YOLOv12 detection
            yolo_label = f"YOLOv12: {fish.get('detection', {}).get('detected_class', 'Unknown')}"
            cv2.putText(image, yolo_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Species classification
            species_label = f"Species: {fish.get('classification', {}).get('species', 'Unknown')}"
            cv2.putText(image, species_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Confidence scores
            yolo_conf = fish.get('detection', {}).get('confidence', 0)
            class_conf = fish.get('classification', {}).get('accuracy', 0)
            conf_label = f"Conf: YOLOv12={yolo_conf:.2f}, Class={class_conf:.2f}"
            cv2.putText(image, conf_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Size category
            size_cat = fish.get('metrics', {}).get('size_category', 'Unknown')
            cv2.putText(image, f"Size: {size_cat}", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join("static", filename)
        os.makedirs("static", exist_ok=True)
        cv2.imwrite(output_path, image)
        
        return filename
        
    except Exception as e:
        print(f"Annotation error: {e}")
        return None

def analyze_image_comprehensive(image_path):
    """Perform comprehensive fish analysis on uploaded image."""
    start_time = time.time()
    
    try:
        # Check if detector is available
        if not YOLO_AVAILABLE or yolo_detector is None:
            return {
                "success": False,
                "error": "YOLOv12 detector not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "error": "Could not load image",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get image info
        img_height, img_width = image.shape[:2]
        
        # YOLOv12 Detection
        detection_start = time.time()
        detections = yolo_detector.predict(image)
        detection_time = time.time() - detection_start
        
        if not detections or not detections[0]:
            return {
                "success": True,
                "fish_count": 0,
                "message": "No fish detected in the image",
                "image_info": {
                    "width": img_width,
                    "height": img_height,
                    "channels": image.shape[2] if len(image.shape) == 3 else 1
                },
                "processing_time": {
                    "detection_ms": round(detection_time * 1000, 2),
                    "total_ms": round((time.time() - start_time) * 1000, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Process each detected fish
        analysis_results = []
        classification_start = time.time()
        
        for i, fish in enumerate(detections[0]):
            fish_start = time.time()
            
            # Basic detection info
            box = fish.get_box()
            confidence = fish.get_score()
            yolo_class = fish.get_class_name()
            
            # Save fish crop for classification
            fish_crop = fish.get_mask_BGR()
            crop_path = f'temp_analysis_fish_{i}_{uuid.uuid4().hex[:8]}.jpg'
            cv2.imwrite(crop_path, fish_crop)
            
            # Classification
            classification_result = classify_fish_crop(crop_path)
            
            # Enhanced polygon
            polygon = create_enhanced_polygon(crop_path, box)
            
            # Fish metrics
            metrics = calculate_fish_metrics(box, image.shape)
            
            # Compile fish analysis
            fish_analysis = {
                "fish_id": i + 1,
                "detection": {
                    "detected_class": yolo_class,
                    "confidence": round(confidence, 4),
                    "bounding_box": [int(x) for x in box],
                    "model": "YOLOv12-Local-Trained"
                },
                "classification": classification_result,
                "polygon": polygon,
                "metrics": metrics,
                "processing_time_ms": round((time.time() - fish_start) * 1000, 2)
            }
            
            analysis_results.append(fish_analysis)
            
            # Clean up crop file
            try:
                os.remove(crop_path)
            except:
                pass
        
        classification_time = time.time() - classification_start
        
        # Create final analysis result
        final_result = {
            "success": True,
            "fish_count": len(analysis_results),
            "detections": analysis_results,
            "image_info": {
                "width": img_width,
                "height": img_height,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "size_mb": round(os.path.getsize(image_path) / (1024*1024), 2)
            },
            "processing_time": {
                "detection_ms": round(detection_time * 1000, 2),
                "classification_ms": round(classification_time * 1000, 2),
                "total_ms": round((time.time() - start_time) * 1000, 2)
            },
            "models_used": {
                "detection": "YOLOv12-Local-Trained (28 fish species)",
                "classification": "EmbeddingClassifier-v7-1 (426 species)"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Create annotated image
        annotated_filename = create_annotated_image(image_path, final_result)
        if annotated_filename:
            final_result["annotated_image"] = {
                "filename": annotated_filename,
                "url": f"/static/{annotated_filename}"
            }
        
        return final_result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "processing_time": {
                "total_ms": round((time.time() - start_time) * 1000, 2)
            },
            "timestamp": datetime.now().isoformat()
        }

@app.route('/analyze', methods=['POST'])
def analyze_fish():
    """
    Analyze uploaded fish image.
    
    Returns comprehensive analysis including:
    - Fish detection results
    - Species classification
    - Confidence scores
    - Fish metrics (size, position)
    - Annotated image
    - Processing performance
    """
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Use 'image' as the form field name.",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"upload_{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
        
        os.makedirs('uploads', exist_ok=True)
        upload_path = os.path.join("uploads", unique_filename)
        file.save(upload_path)
        
        # Analyze the image
        analysis_result = analyze_image_comprehensive(upload_path)
        
        # Clean up uploaded file
        try:
            os.remove(upload_path)
        except:
            pass
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Fish Analysis API",
        "models": {
            "yolo_v12": "loaded" if YOLO_AVAILABLE else "error",
            "classifier": "available"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": 16,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def model_info():
    """Get information about loaded models."""
    yolo_info = "Not Available"
    if YOLO_AVAILABLE and yolo_detector:
        yolo_info = {
            "model_type": "YOLOv12",
            "training": "Local Custom Training",
            "species_count": 28,
            "species": yolo_detector.class_names,
            "confidence_threshold": yolo_detector.confidence
        }
    
    return jsonify({
        "detection_model": yolo_info,
        "classification_model": {
            "model_type": "EmbeddingClassifier",
            "version": "v7-1",
            "species_count": 426,
            "location": "./classification_rectangle_v7-1/"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/static/<filename>')
def serve_static(filename):
    """Serve static files (annotated images)."""
    return send_from_directory('static', filename)

@app.route('/', methods=['GET'])
def api_info():
    """API information and usage guide."""
    return jsonify({
        "service": "Fish Analysis API",
        "version": "1.0.0",
        "description": "Comprehensive fish detection and analysis using YOLOv12 + Classification",
        "endpoints": {
            "POST /analyze": "Upload image for fish analysis",
            "GET /health": "API health check",
            "GET /models": "Model information",
            "GET /static/<filename>": "Download annotated images"
        },
        "usage": {
            "upload_method": "POST multipart/form-data",
            "field_name": "image",
            "supported_formats": list(ALLOWED_EXTENSIONS),
            "max_size_mb": 16
        },
        "example_curl": "curl -X POST -F 'image=@fish.jpg' http://localhost:5004/analyze",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🐟 Starting Comprehensive Fish Analysis API...")
    print("🔍 Checking models...")
    
    # Model status
    if YOLO_AVAILABLE:
        print("✅ YOLOv12 Detection: Ready")
    else:
        print("❌ YOLOv12 Detection: Error")
    
    # Test classification
    try:
        test_result = subprocess.run([
            sys.executable, "-c", """
import sys, os
os.chdir('./classification_rectangle_v7-1')
from inference import EmbeddingClassifier
classifier = EmbeddingClassifier('./model.ts', './database.pt')
print('CLASS_OK')
"""
        ], capture_output=True, text=True, timeout=30)
        class_ok = "CLASS_OK" in test_result.stdout
        print(f"✅ Classification: {'Ready' if class_ok else 'Error'}")
    except:
        class_ok = False
        print("❌ Classification: Error")
    
    # Create directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n🚀 Starting Fish Analysis API Server...")
    print("📡 API Base URL: http://localhost:5004")
    print("🔍 Analysis Endpoint: POST http://localhost:5004/analyze")
    print("🩺 Health Check: GET http://localhost:5004/health")
    print("📊 Model Info: GET http://localhost:5004/models")
    print("📁 Static Files: GET http://localhost:5004/static/<filename>")
    print("\n📝 Example Usage:")
    print("   curl -X POST -F 'image=@test_image.png' http://localhost:5004/analyze")
    print("   curl http://localhost:5004/health")
    print("\n⏹️  Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5004, debug=True) 
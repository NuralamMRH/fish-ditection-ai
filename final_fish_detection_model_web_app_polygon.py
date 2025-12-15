#!/usr/bin/env python3
"""
Advanced Fish Identification with Interactive Polygons
=====================================================

Enhanced version with:
- Polygon outlines instead of bounding boxes
- Interactive polygon selection
- Detailed results for selected fish
- Professional segmentation display
"""

import os
import sys
import cv2
import json
import subprocess
import tempfile
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, flash, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'fish_polygon_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def test_models():
    """Test if models are working."""
    try:
        # Test YOLO detection
        yolo_test = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
sys.path.insert(0, './detector_v10_m3')
from inference import YOLOInference
detector = YOLOInference('./detector_v10_m3/model.ts')
image = cv2.imread('test_image.png')
detections = detector.predict(image)
if detections and detections[0]:
    print('YOLO_SUCCESS')
else:
    print('YOLO_FAIL')
"""
        ], capture_output=True, text=True, timeout=30)
        
        # Test classification
        class_test = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
os.chdir('./classification_rectangle_v7-1')
from inference import EmbeddingClassifier
classifier = EmbeddingClassifier('./model.ts', './database.pt')
image = cv2.imread('../test_image.png')
results = classifier.inference_numpy(image)
if results:
    print('CLASS_SUCCESS')
else:
    print('CLASS_FAIL')
"""
        ], capture_output=True, text=True, timeout=30)
        
        yolo_ok = "YOLO_SUCCESS" in yolo_test.stdout
        class_ok = "CLASS_SUCCESS" in class_test.stdout
        
        return yolo_ok, class_ok
        
    except Exception as e:
        print(f"Error testing models: {e}")
        return False, False

def mask_to_polygon(mask):
    """Convert a binary mask to polygon coordinates."""
    try:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get the largest contour (main fish body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour to reduce points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of [x, y] coordinates
        polygon_points = []
        for point in simplified_contour:
            x, y = point[0]
            polygon_points.append([int(x), int(y)])
        
        return polygon_points
        
    except Exception as e:
        print(f"Error converting mask to polygon: {e}")
        return []

def create_polygon_annotated_image(image_path, detection_results):
    """Create an annotated image with polygon outlines and clickable areas."""
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        # Define colors for different fish
        colors = [
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta/Pink
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
            (0, 165, 255),    # Orange
            (128, 0, 128),    # Purple
            (255, 0, 0),      # Blue
            (0, 128, 255),    # Red-Orange
        ]
        
        # Create overlay for semi-transparent polygons
        overlay = image.copy()
        
        # Process each fish
        for i, fish in enumerate(detection_results['fish']):
            polygon = fish.get('polygon', [])
            if not polygon:
                continue
                
            species = fish['species']
            accuracy = fish['accuracy']
            fish_id = fish['fish_id']
            
            # Get color for this fish
            color = colors[i % len(colors)]
            
            # Convert polygon to numpy array
            polygon_np = np.array(polygon, dtype=np.int32)
            
            # Draw filled polygon with transparency
            cv2.fillPoly(overlay, [polygon_np], color)
            
            # Draw polygon outline
            cv2.polylines(image, [polygon_np], True, color, 3)
            
            # Calculate centroid for label placement
            M = cv2.moments(polygon_np)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback to polygon center
                cx = int(np.mean([p[0] for p in polygon]))
                cy = int(np.mean([p[1] for p in polygon]))
            
            # Draw fish label
            label = f"#{fish_id}: {species}"
            accuracy_text = f"{accuracy:.1%}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Calculate text size
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (acc_width, acc_height), _ = cv2.getTextSize(accuracy_text, font, font_scale, thickness)
            
            # Background rectangle for text
            bg_width = max(label_width, acc_width) + 20
            bg_height = label_height + acc_height + 20
            
            # Position text background
            text_x = max(10, cx - bg_width // 2)
            text_y = max(bg_height + 10, cy - bg_height // 2)
            
            # Draw text background
            cv2.rectangle(image, 
                         (text_x, text_y - bg_height), 
                         (text_x + bg_width, text_y), 
                         color, -1)
            
            # Draw text
            cv2.putText(image, label, 
                       (text_x + 10, text_y - acc_height - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            cv2.putText(image, accuracy_text, 
                       (text_x + 10, text_y - 5), 
                       font, font_scale, (0, 0, 0), thickness)
        
        # Blend overlay for semi-transparent effect
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Save annotated image
        result_filename = f"polygon_annotated_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(STATIC_FOLDER, result_filename)
        cv2.imwrite(result_path, image)
        
        return result_filename, image.shape
        
    except Exception as e:
        print(f"Error creating polygon annotated image: {e}")
        return None, None

def process_image_with_polygons(image_path):
    """Process image and return results with polygon coordinates."""
    try:
        # First, detect fish with YOLO using the working approach
        yolo_script = f"""
import sys, os, cv2, json
sys.path.insert(0, './detector_v10_m3')
from inference import YOLOInference
detector = YOLOInference('./detector_v10_m3/model.ts')
image = cv2.imread('{image_path}')
detections = detector.predict(image)

if detections and detections[0]:
    fish_data = []
    for i, fish in enumerate(detections[0]):
        box = fish.get_box()
        confidence = fish.get_score()
        
        # Save fish crop for classification
        fish_crop = fish.get_mask_BGR()
        crop_path = f'temp_fish_{{i}}.jpg'
        cv2.imwrite(crop_path, fish_crop)
        
        fish_data.append({{
            'fish_id': i + 1,
            'box': [int(x) for x in box],
            'confidence': float(confidence),
            'crop_path': crop_path
        }})
    
    result = {{'success': True, 'fish': fish_data}}
    print('YOLO_RESULT:' + json.dumps(result))
else:
    print('YOLO_RESULT:' + json.dumps({{'success': False, 'error': 'No fish detected'}}))
"""
        
        yolo_result = subprocess.run([sys.executable, "-c", yolo_script], 
                                   capture_output=True, text=True, timeout=30)
        
        # Parse YOLO result
        yolo_output = None
        for line in yolo_result.stdout.split('\n'):
            if line.startswith('YOLO_RESULT:'):
                yolo_output = json.loads(line.replace('YOLO_RESULT:', ''))
                break
        
        if not yolo_output or not yolo_output.get('success'):
            return {"error": "No fish detected in image"}
        
        # Process each detected fish
        final_results = []
        for fish in yolo_output['fish']:
            crop_path = fish['crop_path']
            
            # Generate polygon from bounding box (simplified approach)
            box = fish['box']
            x1, y1, x2, y2 = box
            
            # Create a simple polygon from bounding box for now
            # This ensures we always have polygon data
            polygon = [
                [x1, y1],           # Top-left
                [x2, y1],           # Top-right
                [x2, y2],           # Bottom-right
                [x1, y2]            # Bottom-left
            ]
            
            # Try to create a more sophisticated polygon from the crop
            try:
                # Load the fish crop and create a more realistic polygon
                crop_image = cv2.imread(crop_path)
                if crop_image is not None:
                    # Convert to grayscale and create a mask
                    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                    
                    # Create a binary mask (simple thresholding)
                    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Simplify the contour
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        # Convert contour points to polygon coordinates in original image space
                        enhanced_polygon = []
                        for point in simplified_contour:
                            # Translate crop coordinates to original image coordinates
                            px, py = point[0]
                            original_x = x1 + px
                            original_y = y1 + py
                            enhanced_polygon.append([int(original_x), int(original_y)])
                        
                        # Only use enhanced polygon if it has reasonable number of points
                        if len(enhanced_polygon) >= 4:
                            polygon = enhanced_polygon
            except Exception as e:
                print(f"Could not enhance polygon for fish {fish['fish_id']}: {e}")
                # Keep the simple rectangle polygon as fallback
            
            # Classify this fish crop
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
        'species_id': results[0]['species_id']
    }}
    print('CLASS_RESULT:' + json.dumps(result))
else:
    print('CLASS_RESULT:' + json.dumps({{'species': 'Unknown', 'accuracy': 0.0}}))
"""
            
            class_result = subprocess.run([sys.executable, "-c", class_script], 
                                        capture_output=True, text=True, timeout=30)
            
            # Parse classification result
            class_output = None
            for line in class_result.stdout.split('\n'):
                if line.startswith('CLASS_RESULT:'):
                    class_output = json.loads(line.replace('CLASS_RESULT:', ''))
                    break
            
            if not class_output:
                class_output = {'species': 'Unknown', 'accuracy': 0.0}
            
            # Combine detection and classification
            final_results.append({
                "fish_id": fish['fish_id'],
                "species": class_output['species'],
                "accuracy": round(class_output['accuracy'], 3),
                "confidence": round(fish['confidence'], 3),
                "box": fish['box'],
                "polygon": polygon
            })
            
            # Clean up temp files
            try:
                os.remove(crop_path)
            except:
                pass
        
        # Create the final result structure
        result = {"success": True, "fish_count": len(final_results), "fish": final_results}
        
        # Create polygon annotated image
        annotated_image, image_shape = create_polygon_annotated_image(image_path, result)
        if annotated_image:
            result["annotated_image"] = annotated_image
            result["image_dimensions"] = {"width": image_shape[1], "height": image_shape[0]}
        
        return result
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Enhanced HTML Template with Interactive Polygons
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐟 Interactive Fish Polygon Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
        }
        .status { 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            background: #e8f5e8; 
            border-left: 4px solid #4caf50;
        }
        .upload-zone { 
            margin: 20px 0; 
            padding: 40px; 
            border: 3px dashed #ddd; 
            border-radius: 12px; 
            text-align: center; 
            transition: all 0.3s ease;
            cursor: pointer;
            background: #fafafa;
        }
        .upload-zone:hover, .upload-zone.dragover { 
            border-color: #2196F3; 
            background: #f0f8ff;
            transform: translateY(-2px);
        }
        .upload-zone.dragover {
            border-color: #4caf50;
            background: #f0fff0;
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
            color: #666;
        }
        .result-container { 
            display: grid; 
            grid-template-columns: 2fr 1fr; 
            gap: 30px; 
            margin: 20px 0; 
        }
        .image-container {
            position: relative;
            text-align: center;
        }
        .interactive-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            cursor: crosshair;
        }
        .polygon-overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        .result-details {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2196F3;
            max-height: 600px;
            overflow-y: auto;
        }
        .fish-item { 
            border: 1px solid #ddd; 
            margin: 10px 0; 
            padding: 15px; 
            border-radius: 8px; 
            background: white;
            border-left: 4px solid #4caf50;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .fish-item:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        .fish-item.selected {
            border-left-color: #ff5722;
            background: #fff3e0;
            box-shadow: 0 4px 12px rgba(255, 87, 34, 0.3);
        }
        .fish-header {
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }
        .species-name {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }
        .accuracy-high { color: #4caf50; }
        .accuracy-medium { color: #ff9800; }
        .accuracy-low { color: #f44336; }
        
        .selected-fish-details {
            margin-top: 20px;
            padding: 20px;
            background: linear-gradient(45deg, #ff5722, #ff7043);
            color: white;
            border-radius: 10px;
            display: none;
        }
        
        button { 
            background: linear-gradient(45deg, #2196F3, #21cbf3);
            color: white; 
            padding: 12px 25px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }
        .error { 
            background: #ffebee; 
            color: #c62828; 
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2196F3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .summary-card {
            background: linear-gradient(45deg, #4caf50, #8bc34a);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .instruction-panel {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #2196F3;
        }
        @media (max-width: 1200px) {
            .result-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 Interactive Fish Polygon Detection</h1>
        <p>Upload fish images and interact with individual fish polygons for detailed analysis</p>
        
        <div class="status">
            <strong>✅ System Status:</strong> Interactive Polygon Detection Ready<br>
            <strong>🔍 YOLOv10 Detection:</strong> {{ yolo_status }}<br>
            <strong>🔬 Species Classification:</strong> {{ class_status }}<br>
            <strong>📊 Database:</strong> 426+ species with polygon segmentation
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="error">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📸</div>
            <h3>Drop fish image here or click to select</h3>
            <p>Supports PNG, JPG, JPEG, GIF, BMP (max 16MB)</p>
            <form id="uploadForm" method="POST" enctype="multipart/form-data" style="display: none;">
                <input type="file" id="fileInput" name="file" accept=".png,.jpg,.jpeg,.gif,.bmp" onchange="handleFileSelect(event)">
            </form>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>🔍 Analyzing fish...</h3>
            <p>Detecting fish and creating interactive polygons...</p>
        </div>
        
        {% if results %}
            {% if results.success %}
                <div class="summary-card">
                    <h2>🎯 Interactive Detection Complete!</h2>
                    <h3>{{ results.fish_count }} Fish Detected with Polygon Outlines</h3>
                </div>
                
                <div class="instruction-panel">
                    <h4>🖱️ How to Interact:</h4>
                    <p>• <strong>Click on any fish polygon</strong> in the image to select it</p>
                    <p>• <strong>Click on fish cards</strong> in the sidebar to highlight them</p>
                    <p>• <strong>Selected fish</strong> will show detailed information</p>
                    <p>• Each fish has a <strong>unique colored polygon</strong> outline</p>
                </div>
                
                <div class="result-container">
                    {% if results.annotated_image %}
                        <div class="image-container">
                            <h3>📷 Interactive Polygon Detection</h3>
                            <img src="/static/{{ results.annotated_image }}" 
                                 alt="Fish Polygon Detection Results" 
                                 class="interactive-image"
                                 id="fishImage"
                                 onclick="handleImageClick(event)">
                            <p><small>Click on any fish polygon to select it for detailed analysis</small></p>
                        </div>
                    {% endif %}
                    
                    <div class="result-details">
                        <h3>🐠 Detected Fish (Click to Select)</h3>
                        
                        {% for fish in results.fish %}
                            <div class="fish-item" 
                                 data-fish-id="{{ fish.fish_id }}"
                                 onclick="selectFish({{ fish.fish_id }})">
                                <div class="fish-header">🐠 Fish #{{ fish.fish_id }}</div>
                                <div class="species-name">{{ fish.species }}</div>
                                <p><strong>Classification Accuracy:</strong> 
                                    <span class="{% if fish.accuracy >= 0.8 %}accuracy-high{% elif fish.accuracy >= 0.6 %}accuracy-medium{% else %}accuracy-low{% endif %}">
                                        {{ (fish.accuracy * 100)|round(1) }}%
                                    </span>
                                </p>
                                <p><strong>Detection Confidence:</strong> {{ (fish.confidence * 100)|round(1) }}%</p>
                                <p><strong>Polygon Points:</strong> {{ fish.polygon|length }} vertices</p>
                            </div>
                        {% endfor %}
                        
                        <div class="selected-fish-details" id="selectedFishDetails">
                            <h4>📊 Selected Fish Details</h4>
                            <div id="selectedFishContent"></div>
                        </div>
                    </div>
                </div>
                
                <script>
                    // Store fish data for interaction
                    const fishData = {{ results.fish | tojson }};
                    let selectedFishId = null;
                    
                    // Point-in-polygon algorithm
                    function pointInPolygon(point, polygon) {
                        let x = point[0], y = point[1];
                        let inside = false;
                        
                        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
                            let xi = polygon[i][0], yi = polygon[i][1];
                            let xj = polygon[j][0], yj = polygon[j][1];
                            
                            if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                                inside = !inside;
                            }
                        }
                        return inside;
                    }
                    
                    // Handle image clicks
                    function handleImageClick(event) {
                        const img = event.target;
                        const rect = img.getBoundingClientRect();
                        
                        // Calculate click coordinates relative to image
                        const scaleX = img.naturalWidth / img.clientWidth;
                        const scaleY = img.naturalHeight / img.clientHeight;
                        
                        const clickX = (event.clientX - rect.left) * scaleX;
                        const clickY = (event.clientY - rect.top) * scaleY;
                        
                        // Check which fish polygon was clicked
                        for (let fish of fishData) {
                            if (fish.polygon && fish.polygon.length > 0) {
                                if (pointInPolygon([clickX, clickY], fish.polygon)) {
                                    selectFish(fish.fish_id);
                                    return;
                                }
                            }
                        }
                        
                        // No fish clicked, deselect
                        deselectAllFish();
                    }
                    
                    // Select a fish
                    function selectFish(fishId) {
                        selectedFishId = fishId;
                        
                        // Update visual selection
                        document.querySelectorAll('.fish-item').forEach(item => {
                            item.classList.remove('selected');
                        });
                        
                        const selectedItem = document.querySelector(`[data-fish-id="${fishId}"]`);
                        if (selectedItem) {
                            selectedItem.classList.add('selected');
                        }
                        
                        // Show detailed information
                        const fish = fishData.find(f => f.fish_id === fishId);
                        if (fish) {
                            showFishDetails(fish);
                        }
                    }
                    
                    // Deselect all fish
                    function deselectAllFish() {
                        selectedFishId = null;
                        document.querySelectorAll('.fish-item').forEach(item => {
                            item.classList.remove('selected');
                        });
                        document.getElementById('selectedFishDetails').style.display = 'none';
                    }
                    
                    // Show detailed fish information
                    function showFishDetails(fish) {
                        const detailsDiv = document.getElementById('selectedFishDetails');
                        const contentDiv = document.getElementById('selectedFishContent');
                        
                        contentDiv.innerHTML = `
                            <h5>🐠 Fish #${fish.fish_id}: ${fish.species}</h5>
                            <p><strong>🎯 Classification Accuracy:</strong> ${(fish.accuracy * 100).toFixed(1)}%</p>
                            <p><strong>📍 Detection Confidence:</strong> ${(fish.confidence * 100).toFixed(1)}%</p>
                            <p><strong>📐 Polygon Vertices:</strong> ${fish.polygon.length} points</p>
                            <p><strong>📊 Bounding Box:</strong> [${fish.box.join(', ')}]</p>
                            <p><strong>🎨 Interactive:</strong> Polygon-based selection</p>
                        `;
                        
                        detailsDiv.style.display = 'block';
                    }
                </script>
                
            {% else %}
                <div class="error">
                    <h2>❌ No Fish Detected</h2>
                    <p>{{ results.error }}</p>
                    <p>Try uploading a clearer image with visible fish.</p>
                </div>
            {% endif %}
        {% endif %}
        
        <div style="margin-top: 30px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
            <h3>🔗 API Integration</h3>
            <p><strong>POST</strong> to <code>/api</code> with 'file' parameter for programmatic access</p>
            <p><strong>Enhanced Features:</strong> Polygon coordinates included in API response</p>
            <p><strong>Example:</strong> <code>curl -X POST -F "file=@fish.jpg" http://localhost:5002/api</code></p>
        </div>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');

        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            uploadZone.classList.add('dragover');
        }

        function unhighlight(e) {
            uploadZone.classList.remove('dragover');
        }

        uploadZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect({ target: { files: files } });
            }
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                // Show loading
                loading.style.display = 'block';
                uploadZone.style.display = 'none';
                
                // Submit form
                uploadForm.submit();
            }
        }

        // Auto-scroll to results if they exist
        {% if results %}
            setTimeout(() => {
                const resultsSection = document.querySelector('.summary-card');
                if (resultsSection) {
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                }
            }, 500);
        {% endif %}
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    # Check model status
    yolo_ok, class_ok = test_models()
    yolo_status = "✅ Working" if yolo_ok else "❌ Error"
    class_status = "✅ Working" if class_ok else "❌ Error"
    
    results = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            if not (yolo_ok and class_ok):
                flash('Models not ready. Please check console for errors.')
                return redirect(request.url)
                
            try:
                # Save uploaded file temporarily
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = secure_filename(file.filename)
                temp_filename = f"{timestamp}_{filename}"
                temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                file.save(temp_path)
                
                # Process the image with polygon detection
                results = process_image_with_polygons(temp_path)
                
                # Clean up uploaded file
                os.unlink(temp_path)
                
            except Exception as e:
                results = {"error": f"Server error: {str(e)}"}
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
            return redirect(request.url)
    
    return render_template_string(HTML_TEMPLATE, 
                                results=results, 
                                yolo_status=yolo_status, 
                                class_status=class_status)

@app.route('/static/<filename>')
def static_files(filename):
    """Serve static files (annotated images)."""
    return send_file(os.path.join(STATIC_FOLDER, filename))

@app.route('/api', methods=['POST'])
def api():
    """API endpoint for fish identification with polygon data."""
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return {"error": "Invalid file type"}, 400
    
    # Check if models are ready
    yolo_ok, class_ok = test_models()
    if not (yolo_ok and class_ok):
        return {"error": "Models not ready"}, 503
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            results = process_image_with_polygons(tmp.name)
            os.unlink(tmp.name)
        return results
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}, 500

@app.route('/health')
def health():
    """Health check endpoint."""
    yolo_ok, class_ok = test_models()
    return {
        "status": "healthy" if (yolo_ok and class_ok) else "degraded",
        "yolo_detector": "✅ Working" if yolo_ok else "❌ Error",
        "fish_classifier": "✅ Working" if class_ok else "❌ Error",
        "models_ready": yolo_ok and class_ok,
        "features": ["interactive_polygons", "polygon_selection", "advanced_segmentation", "click_detection"],
        "api_version": "3.0"
    }

if __name__ == '__main__':
    print("🐟 Starting Interactive Fish Polygon Detection System...")
    print("🔍 Testing models...")
    
    yolo_ok, class_ok = test_models()
    
    print(f"📍 YOLO Detection: {'✅ Working' if yolo_ok else '❌ Error'}")
    print(f"🔬 Classification: {'✅ Working' if class_ok else '❌ Error'}")
    
    if not (yolo_ok and class_ok):
        print("\n⚠️  Some models have issues, but starting server anyway...")
        print("   Check the /health endpoint for status")
    else:
        print("\n🎉 All models working perfectly!")
    
    print("\n🚀 Starting interactive polygon detection server...")
    print("📱 Open your browser and go to: http://localhost:5002")
    print("✨ Features: Interactive Polygons + Click Selection + Advanced Segmentation")
    print("🔗 API endpoint: POST to http://localhost:5002/api")
    print("🩺 Health check: http://localhost:5002/health")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    app.run(host='0.0.0.0', port=5002, debug=True) 
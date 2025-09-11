#!/usr/bin/env python3
"""
YOLOv12 Fish Detection Web Application
=====================================

Uses YOLOv12 from Roboflow for detection + existing classification system.
"""

import os
import sys
import cv2
import json
import uuid
import subprocess
import numpy as np
from flask import Flask, render_template_string, request, jsonify, send_from_directory

# Add detector_v12 to path
sys.path.insert(0, './detector_v12')
from roboflow_inference import RoboflowYOLOv12Fish

app = Flask(__name__)

# Initialize YOLOv12 detector
try:
    yolo_detector = RoboflowYOLOv12Fish(confidence=0.3)
    print("✅ YOLOv12 Roboflow detector loaded")
except Exception as e:
    print(f"❌ Failed to load YOLOv12: {e}")
    yolo_detector = None

def process_image_with_yolo12(image_path):
    """Process image using YOLOv12 Roboflow API and classification."""
    try:
        # Check if detector is available
        if yolo_detector is None:
            return {"error": "YOLOv12 detector not available"}
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # YOLOv12 Detection
        detections = yolo_detector.predict(image)
        
        if not detections or not detections[0]:
            return {"error": "No fish detected by YOLOv12"}
        
        final_results = []
        
    for i, fish in enumerate(detections[0]):
        box = fish.get_box()
        confidence = fish.get_score()
            yolo_class = fish.get_class_name()
        
        # Save fish crop for classification
        fish_crop = fish.get_mask_BGR()
            crop_path = f'temp_yolo12_fish_{i}.jpg'
        cv2.imwrite(crop_path, fish_crop)
        
            # Enhanced classification using existing classifier
            classification_result = classify_fish_crop(crop_path)
            
            # Create enhanced polygon from crop
            polygon = create_enhanced_polygon(crop_path, box)
            
            final_results.append({
                "fish_id": i + 1,
                "yolo_detection": yolo_class,
                "yolo_confidence": round(confidence, 3),
                "species": classification_result['species'],
                "accuracy": classification_result['accuracy'],
                "box": [int(x) for x in box],
                "polygon": polygon
            })
            
            # Clean up
            try:
                os.remove(crop_path)
            except:
                pass
        
        # Create result
        result = {
            "success": True, 
            "fish_count": len(final_results), 
            "fish": final_results,
            "model": "YOLOv12-Roboflow"
        }
        
        # Create annotated image
        annotated_image = create_yolo12_annotated_image(image_path, result)
        if annotated_image:
            result["annotated_image"] = annotated_image
            result["image_dimensions"] = {"width": image.shape[1], "height": image.shape[0]}
        
        return result
        
            except Exception as e:
        return {"error": f"YOLOv12 processing failed: {str(e)}"}
            
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
        'species_id': results[0]['species_id']
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
                class_output = {'species': 'Unknown', 'accuracy': 0.0}
            
        return class_output
        
    except Exception as e:
        print(f"Classification error: {e}")
        return {'species': 'Unknown', 'accuracy': 0.0}

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

def create_yolo12_annotated_image(image_path, result):
    """Create annotated image with YOLOv12 detections."""
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
        
        for i, fish in enumerate(result.get("fish", [])):
            color = colors[i % len(colors)]
            polygon = fish.get("polygon", [])
            
            if len(polygon) >= 4:
                # Draw polygon
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Semi-transparent overlay
                overlay = image.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
                
                # Polygon outline
                cv2.polylines(image, [pts], True, color, 2)
            
            # Labels
            box = fish.get("box", [0, 0, 100, 100])
            x1, y1, x2, y2 = box
            
            # YOLOv12 detection label
            yolo_label = f"YOLOv12: {fish.get('yolo_detection', 'Unknown')}"
            cv2.putText(image, yolo_label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Classification label  
            class_label = f"Species: {fish.get('species', 'Unknown')}"
            cv2.putText(image, class_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Confidence
            conf_label = f"Conf: {fish.get('yolo_confidence', 0):.2f}"
            cv2.putText(image, conf_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        filename = f"yolo12_annotated_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join("static", filename)
        cv2.imwrite(output_path, image)
        
        return filename
        
    except Exception as e:
        print(f"Annotation error: {e}")
        return None

# HTML Template for YOLOv12 app
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🐟 YOLOv12 Fish Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: white;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-area {
            border: 3px dashed rgba(255,255,255,0.5); border-radius: 15px;
            padding: 40px; text-align: center; background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px); margin-bottom: 30px; transition: all 0.3s ease;
        }
        .upload-area:hover { border-color: rgba(255,255,255,0.8); background: rgba(255,255,255,0.2); }
        .upload-area.dragover { border-color: #4CAF50; background: rgba(76,175,80,0.2); }
        input[type="file"] { display: none; }
        .upload-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            border: none; color: white; padding: 15px 30px; font-size: 16px;
            border-radius: 25px; cursor: pointer; transition: all 0.3s ease;
        }
        .upload-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .results { 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 20px; margin-top: 20px;
        }
        .fish-item {
            background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px;
            margin: 10px 0; border-left: 4px solid #4CAF50;
        }
        .image-result { text-align: center; margin: 20px 0; }
        .image-result img { max-width: 100%; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .model-badge { 
            display: inline-block; background: #FF6B6B; color: white; 
            padding: 5px 15px; border-radius: 20px; font-size: 12px; font-weight: bold;
        }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 3px solid rgba(255,255,255,0.3); border-top: 3px solid white; 
                  border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; 
                  margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐟 YOLOv12 Fish Detection System</h1>
            <span class="model-badge">YOLOv12 + Roboflow API</span>
            <p>Advanced fish detection with 28 species recognition</p>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <h3>📸 Upload Fish Image</h3>
            <p>Drag & drop an image here, or click to select</p>
            <input type="file" id="fileInput" accept="image/*">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose Image
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>🔍 Analyzing image with YOLOv12...</p>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        // Drag and drop
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
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            loading.style.display = 'block';
            results.innerHTML = '';

            fetch('/api', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                displayResults(data);
            })
            .catch(error => {
                loading.style.display = 'none';
                results.innerHTML = '<div class="results"><h3>❌ Error: ' + error.message + '</h3></div>';
            });
        }

        function displayResults(data) {
            if (data.error) {
                results.innerHTML = '<div class="results"><h3>❌ ' + data.error + '</h3></div>';
                return;
            }

            let html = '<div class="results">';
            html += '<h3>🎯 YOLOv12 Detection Results</h3>';
            html += '<p><strong>Model:</strong> ' + (data.model || 'YOLOv12-Roboflow') + '</p>';
            html += '<p><strong>Fish Detected:</strong> ' + data.fish_count + '</p>';

            if (data.annotated_image) {
                html += '<div class="image-result">';
                html += '<img src="/static/' + data.annotated_image + '" alt="Annotated Result">';
                html += '</div>';
            }

            data.fish.forEach((fish, index) => {
                html += '<div class="fish-item">';
                html += '<h4>🐟 Fish ' + fish.fish_id + '</h4>';
                html += '<p><strong>YOLOv12 Detection:</strong> ' + fish.yolo_detection + 
                        ' (' + (fish.yolo_confidence * 100).toFixed(1) + '%)</p>';
                html += '<p><strong>Species Classification:</strong> ' + fish.species + 
                        ' (' + (fish.accuracy * 100).toFixed(1) + '%)</p>';
                html += '<p><strong>Bounding Box:</strong> [' + fish.box.join(', ') + ']</p>';
                html += '</div>';
            });

            html += '</div>';
            results.innerHTML = html;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api', methods=['POST'])
def api():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"})
        
        # Save uploaded file
        filename = f"upload_{uuid.uuid4().hex[:8]}.jpg"
        upload_path = os.path.join("uploads", filename)
        file.save(upload_path)
        
        # Process with YOLOv12
        result = process_image_with_yolo12(upload_path)
            
            # Clean up
            try:
            os.remove(upload_path)
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/static/<filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model": "YOLOv12-Roboflow",
        "yolo_loaded": yolo_detector is not None
    })

if __name__ == '__main__':
    print("🐟 Starting YOLOv12 Fish Detection System...")
    print("🔍 Testing YOLOv12 models...")
    
    # Test YOLOv12
    if yolo_detector:
        print("✅ YOLOv12 Detection: Working")
        yolo_ok = True
    else:
        print("❌ YOLOv12 Detection: Error")
        yolo_ok = False
    
    # Test classification
    try:
        test_result = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
os.chdir('./classification_rectangle_v7-1')
from inference import EmbeddingClassifier
classifier = EmbeddingClassifier('./model.ts', './database.pt')
print('CLASS_SUCCESS')
"""
        ], capture_output=True, text=True, timeout=30)
        class_ok = "CLASS_SUCCESS" in test_result.stdout
        print(f"✅ Classification: {'Working' if class_ok else 'Error'}")
    except:
        class_ok = False
        print("❌ Classification: Error")
    
    if yolo_ok and class_ok:
        print("\n🎉 All YOLOv12 systems working!")
    else:
        print("\n⚠️  Some models have issues...")
    
    # Create directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n🚀 Starting YOLOv12 server...")
    print("📱 Open: http://localhost:5003")
    print("✨ Features: YOLOv12 Roboflow API + Enhanced Classification")
    print("🔗 API endpoint: POST to http://localhost:5003/api")
    print("🩺 Health check: http://localhost:5003/health")
    print("⏹️  Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5003, debug=True)

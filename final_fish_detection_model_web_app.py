#!/usr/bin/env python3
"""
Enhanced Fish Identification Web App with Visual Results
=======================================================

A web application with drag & drop and visual bounding boxes
showing detected fish with species names overlaid.
"""

import os
import sys
import cv2
import json
import subprocess
import tempfile
import uuid
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, flash, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'fish_secret_2024'

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

def create_annotated_image(image_path, detection_results):
    """Create an annotated image with bounding boxes and species labels."""
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Define colors for different fish (similar to the reference image)
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
        
        # Draw bounding boxes and labels for each fish
        for i, fish in enumerate(detection_results['fish']):
            box = fish['box']
            species = fish['species']
            accuracy = fish['accuracy']
            confidence = fish['confidence']
            
            # Get color for this fish (cycle through colors)
            color = colors[i % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text
            label = f"{species}"
            confidence_text = f"Acc: {accuracy:.2f}"
            
            # Calculate text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
            
            # Use the wider text for background width
            bg_width = max(label_width, conf_width) + 10
            bg_height = label_height + conf_height + 15
            
            # Draw background rectangle for text
            cv2.rectangle(image, 
                         (x1, y1 - bg_height - 5), 
                         (x1 + bg_width, y1), 
                         color, -1)
            
            # Draw species name
            cv2.putText(image, label, 
                       (x1 + 5, y1 - conf_height - 8), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Draw accuracy below species name
            cv2.putText(image, confidence_text, 
                       (x1 + 5, y1 - 3), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Add fish number in top-left corner of bounding box
            fish_number = f"#{fish['fish_id']}"
            cv2.putText(image, fish_number, 
                       (x1 + 5, y1 + 25), 
                       font, 0.5, color, 2)
        
        # Save annotated image
        result_filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(STATIC_FOLDER, result_filename)
        cv2.imwrite(result_path, image)
        
        return result_filename
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return None

def process_image_external(image_path):
    """Process image using external scripts to avoid import conflicts."""
    try:
        # First, detect fish with YOLO
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
        
        # Save fish crop
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
        
        # Now classify each detected fish
        final_results = []
        for fish in yolo_output['fish']:
            crop_path = fish['crop_path']
            
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
                "box": fish['box']
            })
            
            # Clean up temp file
            try:
                os.remove(crop_path)
            except:
                pass
        
        # Create the final result structure
        result = {"success": True, "fish_count": len(final_results), "fish": final_results}
        
        # Create annotated image with bounding boxes
        annotated_image = create_annotated_image(image_path, result)
        if annotated_image:
            result["annotated_image"] = annotated_image
        
        return result
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Enhanced HTML Template with Drag & Drop and Visual Results
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐟 Enhanced Fish Identification System</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
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
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin: 20px 0; 
        }
        .result-image {
            text-align: center;
        }
        .result-image img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .result-details {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2196F3;
        }
        .fish-item { 
            border: 1px solid #ddd; 
            margin: 10px 0; 
            padding: 15px; 
            border-radius: 8px; 
            background: white;
            border-left: 4px solid #4caf50;
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
        @media (max-width: 768px) {
            .result-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 Enhanced Fish Identification System</h1>
        <p>Upload fish images with drag & drop to identify species and see visual bounding boxes</p>
        
        <div class="status">
            <strong>✅ System Status:</strong> Enhanced Visual Detection Ready<br>
            <strong>🔍 YOLOv10 Detection:</strong> {{ yolo_status }}<br>
            <strong>🔬 Species Classification:</strong> {{ class_status }}<br>
            <strong>📊 Database:</strong> 426+ fish species with visual bounding boxes
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
            <p>Detecting fish and identifying species...</p>
        </div>
        
        {% if results %}
            {% if results.success %}
                <div class="summary-card">
                    <h2>🎯 Detection Complete!</h2>
                    <h3>{{ results.fish_count }} Fish Detected and Identified</h3>
                </div>
                
                <div class="result-container">
                    {% if results.annotated_image %}
                        <div class="result-image">
                            <h3>📷 Visual Detection Results</h3>
                            <img src="/static/{{ results.annotated_image }}" alt="Fish Detection Results">
                            <p><small>Fish are outlined with colored bounding boxes and labeled with species names</small></p>
                        </div>
                    {% endif %}
                    
                    <div class="result-details">
                        <h3>📊 Detailed Identification Results</h3>
                        
                        {% for fish in results.fish %}
                            <div class="fish-item">
                                <div class="fish-header">🐠 Fish #{{ fish.fish_id }}</div>
                                <div class="species-name">{{ fish.species }}</div>
                                <p><strong>Classification Accuracy:</strong> 
                                    <span class="{% if fish.accuracy >= 0.8 %}accuracy-high{% elif fish.accuracy >= 0.6 %}accuracy-medium{% else %}accuracy-low{% endif %}">
                                        {{ (fish.accuracy * 100)|round(1) }}%
                                    </span>
                                </p>
                                <p><strong>Detection Confidence:</strong> {{ (fish.confidence * 100)|round(1) }}%</p>
                                <p><strong>Location:</strong> Box [{{ fish.box[0] }}, {{ fish.box[1] }}, {{ fish.box[2] }}, {{ fish.box[3] }}]</p>
                            </div>
                        {% endfor %}
                    </div>
                </div>
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
            <p><strong>Health Check:</strong> <a href="/health">/health</a></p>
            <p><strong>Example:</strong> <code>curl -X POST -F "file=@fish.jpg" http://localhost:5001/api</code></p>
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
                
                # Process the image
                results = process_image_external(temp_path)
                
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
    """API endpoint for fish identification."""
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
            results = process_image_external(tmp.name)
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
        "features": ["visual_bounding_boxes", "drag_drop", "species_labeling"],
        "api_version": "2.0"
    }

if __name__ == '__main__':
    print("🐟 Starting Enhanced Fish Identification Web App...")
    print("🔍 Testing models...")
    
    yolo_ok, class_ok = test_models()
    
    print(f"📍 YOLO Detection: {'✅ Working' if yolo_ok else '❌ Error'}")
    print(f"🔬 Classification: {'✅ Working' if class_ok else '❌ Error'}")
    
    if not (yolo_ok and class_ok):
        print("\n⚠️  Some models have issues, but starting server anyway...")
        print("   Check the /health endpoint for status")
    else:
        print("\n🎉 All models working perfectly!")
    
    print("\n🚀 Starting enhanced web server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("✨ Features: Drag & Drop + Visual Bounding Boxes + Species Labels")
    print("🔗 API endpoint: POST to http://localhost:5001/api")
    print("🩺 Health check: http://localhost:5001/health")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    app.run(host='0.0.0.0', port=5001, debug=True) 
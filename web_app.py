#!/usr/bin/env python3
"""
Fish Identification Web Application
===================================

A Flask web application for fish identification using local models.
Upload images and get species identification results in real-time.

Features:
- Image upload interface
- YOLO fish detection  
- Species classification (426 species)
- Results display with confidence scores
- Downloadable results

Author: Fish Identification Project
"""

import os
import sys
import cv2
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'fish_identification_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models (loaded once)
detector = None
classifier = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load fish identification models."""
    global detector, classifier
    
    try:
        print("🐟 Loading Fish Identification Models...")
        
        # Load YOLO detector
        sys.path.insert(0, './detector_v10_m3')
        from inference import YOLOInference
        detector = YOLOInference(
            model_path='./detector_v10_m3/model.ts',
            conf_threshold=0.25,
            nms_threshold=0.45,
            yolo_ver='v10'
        )
        sys.path.remove('./detector_v10_m3')
        print("✅ YOLO Detector loaded")
        
        # Load fish classifier
        sys.path.insert(0, './classification_rectangle_v7-1')
        from inference import EmbeddingClassifier
        classifier = EmbeddingClassifier(
            model_path='./classification_rectangle_v7-1/model.ts',
            data_set_path='./classification_rectangle_v7-1/database.pt',
            device='cpu'
        )
        sys.path.remove('./classification_rectangle_v7-1')
        print("✅ Fish Classifier loaded")
        
        print("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def process_fish_image(image_path):
    """Process uploaded image and return fish identification results."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "filename": os.path.basename(image_path),
                "size": image.shape,
                "dimensions": f"{image.shape[1]}x{image.shape[0]}"
            },
            "detections": [],
            "summary": {}
        }
        
        # Step 1: Fish Detection
        detections = detector.predict(image)
        
        if not detections or not detections[0]:
            results["summary"] = {
                "total_fish": 0,
                "message": "No fish detected in image"
            }
            return results
        
        fish_results = detections[0]
        results["summary"]["total_fish"] = len(fish_results)
        
        # Step 2: Process each detected fish
        for i, fish in enumerate(fish_results):
            fish_data = {
                "fish_id": i + 1,
                "detection": {},
                "classification": {}
            }
            
            # Get detection info
            box = fish.get_box()
            confidence = fish.get_score()
            area = fish.get_area()
            
            fish_data["detection"] = {
                "bounding_box": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                },
                "confidence": round(float(confidence), 3),
                "area_pixels": int(area)
            }
            
            # Get fish crop for classification
            fish_crop = fish.get_mask_BGR()
            
            # Save individual fish crop
            fish_filename = f"fish_{i+1}_{uuid.uuid4().hex[:8]}.jpg"
            fish_path = os.path.join(RESULTS_FOLDER, fish_filename)
            cv2.imwrite(fish_path, fish_crop)
            fish_data["crop_image"] = fish_filename
            
            # Step 3: Fish Classification
            classification_results = classifier.inference_numpy(fish_crop)
            
            if classification_results:
                # Get top 3 results
                top_results = []
                for j, result in enumerate(classification_results[:3]):
                    top_results.append({
                        "rank": j + 1,
                        "species_name": result['name'],
                        "species_id": result['species_id'],
                        "accuracy": round(float(result['accuracy']), 3),
                        "times_found": result.get('times', 0)
                    })
                
                fish_data["classification"] = {
                    "primary_species": top_results[0]["species_name"],
                    "primary_accuracy": top_results[0]["accuracy"],
                    "top_results": top_results
                }
            else:
                fish_data["classification"] = {
                    "primary_species": "Unknown",
                    "primary_accuracy": 0.0,
                    "top_results": []
                }
            
            results["detections"].append(fish_data)
        
        # Create annotated result image
        result_image = image.copy()
        for fish_data in results["detections"]:
            box = fish_data["detection"]["bounding_box"]
            species = fish_data["classification"]["primary_species"]
            accuracy = fish_data["classification"]["primary_accuracy"]
            
            # Draw bounding box
            cv2.rectangle(result_image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{species} ({accuracy:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (box["x1"], box["y1"] - label_size[1] - 10), 
                         (box["x1"] + label_size[0], box["y1"]), (0, 255, 0), -1)
            cv2.putText(result_image, label, (box["x1"], box["y1"] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save annotated result
        result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)
        results["annotated_image"] = result_filename
        
        return results
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process image."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process image
        results = process_fish_image(file_path)
        
        # Save results to JSON
        results_filename = f"results_{timestamp}.json"
        results_path = os.path.join(RESULTS_FOLDER, results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['results_file'] = results_filename
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return render_template('results.html', results=results)
    
    flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
    return redirect(request.url)

@app.route('/api/identify', methods=['POST'])
def api_identify():
    """API endpoint for fish identification."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        results = process_fish_image(tmp.name)
        os.unlink(tmp.name)
    
    return jsonify(results)

@app.route('/results/<filename>')
def download_file(filename):
    """Download result files."""
    return send_file(os.path.join(RESULTS_FOLDER, filename))

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": detector is not None and classifier is not None,
        "timestamp": datetime.now().isoformat()
    })

# Create HTML templates directory
template_dir = 'templates'
os.makedirs(template_dir, exist_ok=True)

# Create index.html template
index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐟 Fish Identification System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f8ff; }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .file-input { margin: 20px 0; }
        .submit-btn { background: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .submit-btn:hover { background: #1976D2; }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .feature { background: white; padding: 20px; border-radius: 8px; text-align: center; }
        .api-info { background: #e3f2fd; padding: 20px; border-radius: 8px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐟 Fish Identification System</h1>
        <p>Upload a fish image to identify species with AI-powered recognition</p>
    </div>

    <form class="upload-form" method="POST" action="/upload" enctype="multipart/form-data">
        <h2>Upload Fish Image</h2>
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div style="color: red; margin: 10px 0;">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="file-input">
            <label for="file">Choose Image File:</label><br>
            <input type="file" id="file" name="file" accept=".png,.jpg,.jpeg,.gif,.bmp" required>
        </div>
        
        <button type="submit" class="submit-btn">🔍 Identify Fish</button>
    </form>

    <div class="features">
        <div class="feature">
            <h3>🎯 Advanced Detection</h3>
            <p>YOLOv10 model finds fish in complex images</p>
        </div>
        <div class="feature">
            <h3>🔬 Species Classification</h3>
            <p>Identifies from 426+ fish species</p>
        </div>
        <div class="feature">
            <h3>📊 Detailed Results</h3>
            <p>Confidence scores and bounding boxes</p>
        </div>
    </div>

    <div class="api-info">
        <h3>🔗 API Usage</h3>
        <p><strong>POST</strong> to <code>/api/identify</code> with image file</p>
        <p><strong>Health Check:</strong> <a href="/health">/health</a></p>
    </div>
</body>
</html>'''

# Create results.html template
results_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐟 Fish Identification Results</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f0f8ff; }
        .header { text-align: center; margin-bottom: 30px; }
        .results-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .summary { background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .fish-result { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }
        .detection-info { background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .classification-info { background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .top-results { margin-top: 10px; }
        .species-result { padding: 5px 0; border-bottom: 1px solid #eee; }
        .download-links { margin: 20px 0; }
        .download-links a { display: inline-block; margin: 5px 10px; padding: 8px 15px; background: #2196F3; color: white; text-decoration: none; border-radius: 5px; }
        .back-link { text-align: center; margin-top: 30px; }
        .error { background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐟 Fish Identification Results</h1>
    </div>

    <div class="results-container">
        {% if results.error %}
            <div class="error">
                <h2>❌ Error</h2>
                <p>{{ results.error }}</p>
            </div>
        {% else %}
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Image:</strong> {{ results.image_info.filename }}</p>
                <p><strong>Dimensions:</strong> {{ results.image_info.dimensions }}</p>
                <p><strong>Fish Detected:</strong> {{ results.summary.total_fish }}</p>
                <p><strong>Processed:</strong> {{ results.timestamp }}</p>
            </div>

            {% if results.summary.total_fish > 0 %}
                {% if results.annotated_image %}
                    <div class="download-links">
                        <h3>📸 Results</h3>
                        <a href="/results/{{ results.annotated_image }}" target="_blank">View Annotated Image</a>
                        <a href="/results/{{ results.results_file }}" download>Download JSON Results</a>
                    </div>
                {% endif %}

                {% for fish in results.detections %}
                    <div class="fish-result">
                        <h3>🐠 Fish #{{ fish.fish_id }}</h3>
                        
                        <div class="detection-info">
                            <h4>📍 Detection Information</h4>
                            <p><strong>Confidence:</strong> {{ fish.detection.confidence }}</p>
                            <p><strong>Area:</strong> {{ fish.detection.area_pixels }} pixels</p>
                            <p><strong>Bounding Box:</strong> ({{ fish.detection.bounding_box.x1 }}, {{ fish.detection.bounding_box.y1 }}) to ({{ fish.detection.bounding_box.x2 }}, {{ fish.detection.bounding_box.y2 }})</p>
                            {% if fish.crop_image %}
                                <a href="/results/{{ fish.crop_image }}" target="_blank">View Fish Crop</a>
                            {% endif %}
                        </div>

                        <div class="classification-info">
                            <h4>🔬 Species Classification</h4>
                            <p><strong>Primary Species:</strong> {{ fish.classification.primary_species }}</p>
                            <p><strong>Accuracy:</strong> {{ fish.classification.primary_accuracy }}</p>
                            
                            {% if fish.classification.top_results %}
                                <div class="top-results">
                                    <h5>Top Classification Results:</h5>
                                    {% for result in fish.classification.top_results %}
                                        <div class="species-result">
                                            <strong>{{ result.rank }}.</strong> {{ result.species_name }} 
                                            (Accuracy: {{ result.accuracy }}, Species ID: {{ result.species_id }})
                                        </div>
                                    {% endfor %}
                                </div>
                            {% endif %}
                        </div>
                    </div>
                {% endfor %}
            {% else %}
                <div class="summary">
                    <p>{{ results.summary.message }}</p>
                </div>
            {% endif %}
        {% endif %}

        <div class="back-link">
            <a href="/" style="display: inline-block; margin: 20px 0; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">🔄 Upload Another Image</a>
        </div>
    </div>
</body>
</html>'''

# Write template files
with open(os.path.join(template_dir, 'index.html'), 'w') as f:
    f.write(index_html)

with open(os.path.join(template_dir, 'results.html'), 'w') as f:
    f.write(results_html)

if __name__ == '__main__':
    print("🐟 Starting Fish Identification Web Application...")
    
    # Load models
    if not load_models():
        print("❌ Failed to load models. Please check model files and virtual environment.")
        sys.exit(1)
    
    print("🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("💡 For API usage: POST images to http://localhost:5000/api/identify")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 
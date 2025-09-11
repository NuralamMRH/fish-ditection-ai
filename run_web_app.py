#!/usr/bin/env python3
"""
Simple Fish Identification Web App
==================================

Easy-to-run web application for fish identification.
Just run this script and open your browser!
"""

import os
import sys
import cv2
import json
from flask import Flask, request, render_template_string, redirect, flash, send_file
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'fish_secret_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Models
detector = None
classifier = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load fish identification models."""
    global detector, classifier
    
    try:
        print("🐟 Loading Fish Identification Models...")
        
        # Load YOLO detector
        print("📍 Loading YOLOv10 Fish Detector...")
        sys.path.insert(0, './detector_v10_m3')
        try:
            from inference import YOLOInference
            detector = YOLOInference('./detector_v10_m3/model.ts', conf_threshold=0.25)
            print("✅ YOLO Detector loaded successfully")
        finally:
            # Always remove the path
            if './detector_v10_m3' in sys.path:
                sys.path.remove('./detector_v10_m3')
        
        # Load fish classifier
        print("🔬 Loading Fish Classifier...")
        original_dir = os.getcwd()
        try:
            os.chdir('./classification_rectangle_v7-1')
            # Import from current directory (classification model)
            if '.' not in sys.path:
                sys.path.insert(0, '.')
            from inference import EmbeddingClassifier
            classifier = EmbeddingClassifier('./model.ts', './database.pt', device='cpu')
            print("✅ Fish Classifier loaded successfully")
        finally:
            # Always return to original directory
            os.chdir(original_dir)
            # Clean up sys.path
            if '.' in sys.path:
                sys.path.remove('.')
        
        print("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_image(image_path):
    """Process image and return results."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        print(f"📷 Processing image: {image.shape}")
        
        # Detect fish
        detections = detector.predict(image)
        
        if not detections or not detections[0]:
            return {"error": "No fish detected in image"}
        
        print(f"🐠 Found {len(detections[0])} fish")
        
        results = []
        for i, fish in enumerate(detections[0]):
            # Get detection info
            box = fish.get_box()
            confidence = fish.get_score()
            
            # Get fish crop and classify
            fish_crop = fish.get_mask_BGR()
            
            print(f"🔍 Classifying fish #{i+1}...")
            classification = classifier.inference_numpy(fish_crop)
            
            if classification and len(classification) > 0:
                species = classification[0]['name']
                accuracy = classification[0]['accuracy']
                print(f"✅ Identified: {species} (accuracy: {accuracy:.3f})")
            else:
                species = "Unknown"
                accuracy = 0.0
                print("❓ Species: Unknown")
            
            results.append({
                "fish_id": i + 1,
                "species": species,
                "accuracy": round(accuracy, 3),
                "confidence": round(confidence, 3),
                "box": [int(x) for x in box]
            })
        
        return {"success": True, "fish_count": len(results), "fish": results}
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Processing failed: {str(e)}"}

# HTML Templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐟 Fish Identification</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f8ff; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-form { margin: 20px 0; }
        .result { margin: 20px 0; padding: 15px; background: #e8f5e8; border-radius: 8px; }
        .error { background: #ffebee; color: #c62828; }
        .fish-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        button { background: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976D2; }
        input[type="file"] { margin: 10px 0; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; background: #fff3cd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 Fish Identification System</h1>
        <p>Upload a fish image to identify the species using AI</p>
        
        <div class="status">
            <strong>Status:</strong> ✅ YOLOv10 Detection & Classification Ready<br>
            <strong>Models:</strong> 426+ species database loaded<br>
            <strong>Features:</strong> Real-time detection and classification
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="result error">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form class="upload-form" method="POST" enctype="multipart/form-data">
            <div>
                <label>Choose fish image:</label><br>
                <input type="file" name="file" accept=".png,.jpg,.jpeg,.gif,.bmp" required>
            </div>
            <button type="submit">🔍 Identify Fish</button>
        </form>
        
        {% if results %}
            {% if results.success %}
                <div class="result">
                    <h2>📊 Results</h2>
                    <p><strong>Fish detected:</strong> {{ results.fish_count }}</p>
                    
                    {% for fish in results.fish %}
                        <div class="fish-item">
                            <h3>🐠 Fish #{{ fish.fish_id }}</h3>
                            <p><strong>Species:</strong> {{ fish.species }}</p>
                            <p><strong>Classification Accuracy:</strong> {{ fish.accuracy }}</p>
                            <p><strong>Detection Confidence:</strong> {{ fish.confidence }}</p>
                            <p><strong>Location:</strong> Box {{ fish.box }}</p>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="result error">
                    <h2>❌ Error</h2>
                    <p>{{ results.error }}</p>
                </div>
            {% endif %}
        {% endif %}
        
        <div style="margin-top: 30px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
            <h3>ℹ️ How it works</h3>
            <p>1. <strong>Upload:</strong> Choose a fish image from your device</p>
            <p>2. <strong>Detect:</strong> YOLOv10 finds fish in the image</p>
            <p>3. <strong>Classify:</strong> AI identifies species from 426+ fish types</p>
            <p>4. <strong>Results:</strong> Get species name and confidence scores</p>
            
            <h3>🔗 API Endpoint</h3>
            <p><strong>POST</strong> to <code>/api</code> with 'file' parameter for programmatic access</p>
            <p><strong>Example:</strong> <code>curl -X POST -F "file=@fish.jpg" http://localhost:5000/api</code></p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
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
            # Save and process file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    file.save(tmp.name)
                    results = process_image(tmp.name)
                    os.unlink(tmp.name)
            except Exception as e:
                results = {"error": f"Server error: {str(e)}"}
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
            return redirect(request.url)
    
    return render_template_string(HTML_TEMPLATE, results=results)

@app.route('/api', methods=['POST'])
def api():
    """Simple API endpoint for programmatic access."""
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return {"error": "Invalid file type"}, 400
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            results = process_image(tmp.name)
            os.unlink(tmp.name)
        return results
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}, 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": detector is not None and classifier is not None,
        "yolo_detector": "✅ Loaded" if detector is not None else "❌ Not loaded",
        "fish_classifier": "✅ Loaded" if classifier is not None else "❌ Not loaded",
        "api_version": "1.0"
    }

if __name__ == '__main__':
    print("🐟 Starting Fish Identification Web App...")
    
    if not load_models():
        print("\n❌ Failed to load models. Please check:")
        print("   1. Virtual environment is activated: source venv/bin/activate")
        print("   2. Dependencies installed: pip install -r requirements.txt")
        print("   3. Model files exist in detector_v10_m3/ and classification_rectangle_v7-1/")
        print("\n💡 Try running: python simple_demo.py first to test models")
        sys.exit(1)
    
    print("\n🎉 Models loaded successfully!")
    print("🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔗 API endpoint: POST to http://localhost:5000/api")
    print("🩺 Health check: http://localhost:5000/health")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True) 
#!/usr/bin/env python3
"""
Simple Fish Identification API Demo
===================================

A simple Flask API for fish identification.
Upload images via POST request and get JSON results.
"""

import os
import sys
import cv2
import json
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)

# Global variables for models
detector = None
classifier = None

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
        original_dir = os.getcwd()
        os.chdir('./classification_rectangle_v7-1')
        
        from inference import EmbeddingClassifier
        classifier = EmbeddingClassifier(
            model_path='./model.ts',
            data_set_path='./database.pt',
            device='cpu'
        )
        
        os.chdir(original_dir)
        print("✅ Fish Classifier loaded")
        
        print("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def identify_fish(image_path):
    """Identify fish in image and return results."""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        results = {
            "success": True,
            "image_info": {
                "filename": os.path.basename(image_path),
                "dimensions": f"{image.shape[1]}x{image.shape[0]}"
            },
            "fish_detected": [],
            "total_fish": 0
        }
        
        # Step 1: Fish Detection
        detections = detector.predict(image)
        
        if not detections or not detections[0]:
            results["total_fish"] = 0
            results["message"] = "No fish detected in image"
            return results
        
        fish_results = detections[0]
        results["total_fish"] = len(fish_results)
        
        # Step 2: Process each detected fish
        for i, fish in enumerate(fish_results):
            # Get detection info
            box = fish.get_box()
            confidence = fish.get_score()
            
            # Get fish crop for classification
            fish_crop = fish.get_mask_BGR()
            
            # Classify fish
            classification_results = classifier.inference_numpy(fish_crop)
            
            fish_data = {
                "fish_id": i + 1,
                "detection_confidence": round(float(confidence), 3),
                "bounding_box": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                }
            }
            
            if classification_results:
                top_result = classification_results[0]
                fish_data["species"] = {
                    "name": top_result['name'],
                    "species_id": top_result['species_id'],
                    "accuracy": round(float(top_result['accuracy']), 3)
                }
                
                # Add top 3 results
                fish_data["top_matches"] = []
                for j, result in enumerate(classification_results[:3]):
                    fish_data["top_matches"].append({
                        "rank": j + 1,
                        "species": result['name'],
                        "accuracy": round(float(result['accuracy']), 3)
                    })
            else:
                fish_data["species"] = {
                    "name": "Unknown",
                    "species_id": None,
                    "accuracy": 0.0
                }
                fish_data["top_matches"] = []
            
            results["fish_detected"].append(fish_data)
        
        return results
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "success": False}

@app.route('/')
def home():
    """API information page."""
    return jsonify({
        "message": "🐟 Fish Identification API",
        "endpoints": {
            "POST /identify": "Upload image for fish identification",
            "GET /health": "Check API health",
            "GET /": "This information"
        },
        "usage": "Send POST request to /identify with 'image' file in form data",
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "gif"],
        "models": {
            "detector": "YOLOv10 Fish Detection",
            "classifier": "ConvNeXt Fish Classification (426 species)"
        }
    })

@app.route('/identify', methods=['POST'])
def identify():
    """Fish identification endpoint."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided", "success": False}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected", "success": False}), 400
    
    # Save temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            results = identify_fish(tmp.name)
            os.unlink(tmp.name)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}", "success": False}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": detector is not None and classifier is not None,
        "api_version": "1.0"
    })

if __name__ == '__main__':
    print("🐟 Starting Fish Identification API...")
    
    # Load models
    if not load_models():
        print("❌ Failed to load models. Exiting.")
        sys.exit(1)
    
    print("🚀 API server starting...")
    print("📱 API available at: http://localhost:5000")
    print("📖 Documentation: http://localhost:5000")
    print("🔍 Test endpoint: POST to http://localhost:5000/identify")
    print()
    print("Example usage:")
    print("curl -X POST -F 'image=@your_fish_image.jpg' http://localhost:5000/identify")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 
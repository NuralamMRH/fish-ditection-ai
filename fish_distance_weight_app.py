#!/usr/bin/env python3
"""
Advanced Fish Detection with Distance & Weight Calculation
========================================================

Web application that detects fish, estimates distance using depth estimation,
and calculates fish weight based on length, girth, and distance measurements.
"""

import cv2
import numpy as np
import json
import time
import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp
import math

# Add detector paths
sys.path.insert(0, './detector_v12')
from local_inference import LocalYOLOv12Fish

# Add classification path
sys.path.insert(0, './classification_rectangle_v7-1')
from inference_class import EmbeddingClassifier

app = Flask(__name__)
CORS(app)

class FishDistanceWeightAnalyzer:
    """Advanced fish analyzer with distance and weight calculation."""
    
    def __init__(self):
        """Initialize fish analyzer with all components."""
        
        # Initialize YOLOv12 detector
        try:
            self.detector = LocalYOLOv12Fish(confidence=0.4)
            print("✅ YOLOv12 detector loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLOv12: {e}")
            self.detector = None
        
        # Initialize classifier
        try:
            self.classifier = EmbeddingClassifier()
            print("✅ Fish classifier loaded")
        except Exception as e:
            print(f"❌ Failed to load classifier: {e}")
            self.classifier = None
        
        # Initialize MediaPipe for depth estimation
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe face mesh for reference distance calculation
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera calibration parameters (adjust based on your camera)
        self.camera_focal_length = 800  # pixels (typical for HD cameras)
        self.real_iris_size = 11.7  # mm (average human iris diameter)
        
        # Fish weight calculation constants
        self.weight_formulas = {
            'girth_based': lambda length, girth: (length * girth * girth) / 800,
            'length_based': lambda length: (length * length * length) / 1200,
            'species_specific': {
                'Tuna': lambda length: (length ** 2.8) / 1000,
                'Bass': lambda length: (length ** 3.0) / 1300,
                'Salmon': lambda length: (length ** 2.9) / 1100,
                'Catfish': lambda length: (length ** 3.1) / 1400,
                'Snapper': lambda length: (length ** 2.85) / 1050,
                'Mackerel': lambda length: (length ** 2.7) / 950,
                'Grouper': lambda length: (length ** 3.2) / 1500
            }
        }
        
        # Fish measurement types by species
        self.measurement_types = {
            'total_length': ['Tuna', 'Bass', 'Catfish', 'Salmon', 'Snapper', 'Grouper', 'Freshwater-Eel'],
            'fork_length': ['Mackerel', 'Pompano', 'Blue marlin', 'Vietnamese mackerel', 'Phu Quoc Island Tuna'],
            'lower_jaw_fork': ['marlin', 'sailfish', 'swordfish']
        }
        
    def estimate_distance_from_reference(self, image, fish_bbox):
        """Estimate distance using face detection as reference."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get iris landmarks for distance calculation
                left_iris = []
                right_iris = []
                
                # Left iris landmarks (468-477)
                for idx in range(468, 474):
                    landmark = face_landmarks.landmark[idx]
                    left_iris.append([landmark.x * image.shape[1], landmark.y * image.shape[0]])
                
                # Right iris landmarks (473-478)  
                for idx in range(473, 479):
                    landmark = face_landmarks.landmark[idx]
                    right_iris.append([landmark.x * image.shape[1], landmark.y * image.shape[0]])
                
                if left_iris:
                    # Calculate iris width in pixels
                    iris_points = np.array(left_iris)
                    iris_width_pixels = np.max(iris_points[:, 0]) - np.min(iris_points[:, 0])
                    
                    # Calculate distance using similar triangles
                    # Distance = (real_size * focal_length) / pixel_size
                    face_distance = (self.real_iris_size * self.camera_focal_length) / iris_width_pixels
                    
                    return face_distance  # in mm
        
        # Fallback: estimate based on fish size in frame
        fish_x1, fish_y1, fish_x2, fish_y2 = fish_bbox
        fish_area = (fish_x2 - fish_x1) * (fish_y2 - fish_y1)
        frame_area = image.shape[0] * image.shape[1]
        area_ratio = fish_area / frame_area
        
        # Empirical distance estimation based on fish size in frame
        estimated_distance = 500 / math.sqrt(area_ratio)  # mm
        return estimated_distance
    
    def calculate_fish_dimensions(self, fish_bbox, distance_mm, image_shape):
        """Calculate real fish dimensions from bounding box and distance."""
        x1, y1, x2, y2 = fish_bbox
        
        # Fish dimensions in pixels
        fish_width_pixels = x2 - x1
        fish_height_pixels = y2 - y1
        
        # Convert to real dimensions using distance
        # Real size = (pixel_size * distance) / focal_length
        fish_length_mm = (fish_width_pixels * distance_mm) / self.camera_focal_length
        fish_height_mm = (fish_height_pixels * distance_mm) / self.camera_focal_length
        
        return {
            'length_mm': fish_length_mm,
            'height_mm': fish_height_mm,
            'length_cm': fish_length_mm / 10,
            'height_cm': fish_height_mm / 10,
            'length_inches': fish_length_mm / 25.4,
            'height_inches': fish_height_mm / 25.4
        }
    
    def estimate_fish_girth(self, dimensions, species=None):
        """Estimate fish girth based on length and species."""
        length_cm = dimensions['length_cm']
        
        # Species-specific girth ratios (girth/length)
        girth_ratios = {
            'Tuna': 0.65,
            'Bass': 0.45,
            'Salmon': 0.40,
            'Catfish': 0.55,
            'Snapper': 0.42,
            'Mackerel': 0.35,
            'Grouper': 0.50,
            'default': 0.45
        }
        
        # Get species-specific ratio or use default
        ratio = girth_ratios.get(species, girth_ratios['default'])
        estimated_girth = length_cm * ratio
        
        return estimated_girth
    
    def calculate_fish_weight(self, dimensions, species=None):
        """Calculate fish weight using multiple methods."""
        length_cm = dimensions['length_cm']
        
        # Estimate girth
        girth_cm = self.estimate_fish_girth(dimensions, species)
        
        # Method 1: Girth-based formula
        weight_girth = self.weight_formulas['girth_based'](length_cm, girth_cm)
        
        # Method 2: Length-based formula
        weight_length = self.weight_formulas['length_based'](length_cm)
        
        # Method 3: Species-specific formula
        weight_species = None
        if species and species in self.weight_formulas['species_specific']:
            weight_species = self.weight_formulas['species_specific'][species](length_cm)
        
        # Average the methods for final estimate
        weights = [w for w in [weight_girth, weight_length, weight_species] if w is not None]
        estimated_weight = sum(weights) / len(weights) if weights else 0
        
        return {
            'weight_grams': estimated_weight,
            'weight_kg': estimated_weight / 1000,
            'weight_pounds': estimated_weight * 0.00220462,
            'weight_ounces': estimated_weight * 0.035274,
            'methods': {
                'girth_based': weight_girth,
                'length_based': weight_length,
                'species_specific': weight_species
            },
            'estimated_girth_cm': girth_cm
        }
    
    def get_measurement_type(self, species):
        """Get the appropriate measurement type for the species."""
        for measurement_type, species_list in self.measurement_types.items():
            if any(sp.lower() in species.lower() for sp in species_list):
                return measurement_type
        return 'total_length'  # default
    
    def analyze_image_with_distance_weight(self, image_path):
        """Complete analysis with fish detection, distance estimation, and weight calculation."""
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not load image'}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': {},
            'fish_detections': [],
            'total_fish': 0,
            'distance_estimation': {},
            'weight_calculations': {}
        }
        
        # Step 1: Fish Detection
        detection_start = time.time()
        detections = []
        if self.detector:
            det_results = self.detector.predict(image)
            if det_results and det_results[0]:
                detections = det_results[0]
        
        results['processing_time']['detection'] = time.time() - detection_start
        results['total_fish'] = len(detections)
        
        if not detections:
            results['processing_time']['total'] = time.time() - start_time
            return results
        
        # Step 2: Process each fish
        for i, fish in enumerate(detections):
            fish_start = time.time()
            
            # Get fish detection info
            bbox = fish.get_box()
            confidence = fish.get_score()
            fish_class = fish.get_class_name()
            
            # Step 3: Estimate distance
            distance_mm = self.estimate_distance_from_reference(image, bbox)
            
            # Step 4: Calculate real dimensions
            dimensions = self.calculate_fish_dimensions(bbox, distance_mm, image.shape)
            
            # Step 5: Classify fish species (if classifier available)
            species_classification = None
            if self.classifier:
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    fish_crop = image[y1:y2, x1:x2]
                    if fish_crop.size > 0:
                        species_result = self.classifier.classify_fish(fish_crop)
                        if species_result:
                            species_classification = species_result
                except Exception as e:
                    print(f"Classification error: {e}")
            
            # Determine species for weight calculation
            species_name = None
            if species_classification:
                species_name = species_classification.get('predicted_species', fish_class)
            else:
                species_name = fish_class
            
            # Step 6: Calculate weight
            weight_info = self.calculate_fish_weight(dimensions, species_name)
            
            # Step 7: Get measurement type
            measurement_type = self.get_measurement_type(species_name)
            
            # Compile fish information
            fish_info = {
                'fish_id': i + 1,
                'detection': {
                    'bbox': bbox,
                    'confidence': confidence,
                    'yolo_class': fish_class
                },
                'distance': {
                    'distance_mm': round(distance_mm, 1),
                    'distance_cm': round(distance_mm / 10, 1),
                    'distance_inches': round(distance_mm / 25.4, 2),
                    'estimation_method': 'face_reference' if distance_mm < 1000 else 'size_based'
                },
                'dimensions': dimensions,
                'species_classification': species_classification,
                'weight_calculation': weight_info,
                'measurement_type': measurement_type,
                'processing_time': time.time() - fish_start
            }
            
            results['fish_detections'].append(fish_info)
        
        # Summary statistics
        if results['fish_detections']:
            total_weight = sum(fish['weight_calculation']['weight_grams'] for fish in results['fish_detections'])
            avg_distance = sum(fish['distance']['distance_cm'] for fish in results['fish_detections']) / len(results['fish_detections'])
            
            results['summary'] = {
                'total_estimated_weight': {
                    'grams': round(total_weight, 1),
                    'kg': round(total_weight / 1000, 3),
                    'pounds': round(total_weight * 0.00220462, 2)
                },
                'average_distance_cm': round(avg_distance, 1),
                'largest_fish': max(results['fish_detections'], key=lambda x: x['weight_calculation']['weight_grams']),
                'closest_fish': min(results['fish_detections'], key=lambda x: x['distance']['distance_cm'])
            }
        
        results['processing_time']['total'] = time.time() - start_time
        return results

# Initialize analyzer
analyzer = FishDistanceWeightAnalyzer()

@app.route('/')
def index():
    """Main page with fish distance & weight analyzer."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐟 Fish Distance & Weight Analyzer</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f8ff; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .upload-section { background: white; padding: 20px; border-radius: 10px; 
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .results-section { background: white; padding: 20px; border-radius: 10px; 
                             box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: none; }
            .fish-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; 
                        margin: 10px 0; background: #f9f9f9; }
            .fish-header { display: flex; justify-content: space-between; align-items: center; 
                          margin-bottom: 10px; }
            .fish-title { font-size: 18px; font-weight: bold; color: #2c3e50; }
            .fish-confidence { background: #27ae60; color: white; padding: 4px 8px; 
                             border-radius: 4px; font-size: 12px; }
            .measurements { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                          gap: 15px; margin: 15px 0; }
            .measurement-group { background: white; padding: 12px; border-radius: 6px; 
                               border-left: 4px solid #3498db; }
            .measurement-title { font-weight: bold; color: #2c3e50; margin-bottom: 8px; }
            .measurement-value { font-size: 14px; color: #7f8c8d; margin: 2px 0; }
            .loading { text-align: center; padding: 20px; }
            .error { color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 5px; }
            .success { color: #27ae60; background: #d5f4e6; padding: 10px; border-radius: 5px; }
            button { background: #3498db; color: white; border: none; padding: 12px 24px; 
                    border-radius: 6px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2980b9; }
            input[type="file"] { margin: 10px 0; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                          gap: 20px; margin: 20px 0; }
            .feature-card { background: white; padding: 20px; border-radius: 10px; 
                          box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
            .feature-icon { font-size: 48px; margin-bottom: 10px; }
            .summary-stats { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                        gap: 15px; text-align: center; }
            .stat-item { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; }
            .stat-value { font-size: 24px; font-weight: bold; }
            .stat-label { font-size: 12px; opacity: 0.9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐟 Advanced Fish Distance & Weight Analyzer</h1>
                <p>Upload fish images to detect, measure distance, and calculate weight using AI</p>
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3>Fish Detection</h3>
                    <p>YOLOv12 detects 28+ fish species with high accuracy</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📏</div>
                    <h3>Distance Estimation</h3>
                    <p>MediaPipe-based depth estimation for accurate measurements</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚖️</div>
                    <h3>Weight Calculation</h3>
                    <p>Multiple formulas for precise fish weight estimation</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔬</div>
                    <h3>Species Classification</h3>
                    <p>426+ species database for detailed identification</p>
                </div>
            </div>
            
            <div class="upload-section">
                <h2>📤 Upload Fish Image</h2>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="imageFile" name="image" accept="image/*" required>
                    <br><br>
                    <button type="submit">🔍 Analyze Fish</button>
                </form>
                <div id="loading" class="loading" style="display: none;">
                    <p>🐠 Analyzing fish... Please wait...</p>
                </div>
            </div>
            
            <div id="results" class="results-section"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                formData.append('image', file);
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                try {
                    const response = await fetch('/analyze-distance-weight', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = 
                        '<div class="error">Error analyzing image: ' + error.message + '</div>';
                    document.getElementById('results').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
            
            function displayResults(data) {
                if (data.error) {
                    document.getElementById('results').innerHTML = 
                        '<div class="error">Error: ' + data.error + '</div>';
                    document.getElementById('results').style.display = 'block';
                    return;
                }
                
                let html = '<h2>🐟 Fish Analysis Results</h2>';
                
                // Summary statistics
                if (data.summary) {
                    html += `
                        <div class="summary-stats">
                            <h3>📊 Summary Statistics</h3>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <div class="stat-value">${data.total_fish}</div>
                                    <div class="stat-label">Fish Detected</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">${data.summary.total_estimated_weight.kg} kg</div>
                                    <div class="stat-label">Total Weight</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">${data.summary.total_estimated_weight.pounds} lbs</div>
                                    <div class="stat-label">Total (Pounds)</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">${data.summary.average_distance_cm} cm</div>
                                    <div class="stat-label">Avg Distance</div>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                // Individual fish results
                if (data.fish_detections && data.fish_detections.length > 0) {
                    data.fish_detections.forEach(fish => {
                        html += `
                            <div class="fish-card">
                                <div class="fish-header">
                                    <div class="fish-title">🐠 Fish #${fish.fish_id}</div>
                                    <div class="fish-confidence">${(fish.detection.confidence * 100).toFixed(1)}%</div>
                                </div>
                                
                                <div class="measurements">
                                    <div class="measurement-group">
                                        <div class="measurement-title">🎯 Detection</div>
                                        <div class="measurement-value">Species: ${fish.detection.yolo_class}</div>
                                        <div class="measurement-value">Confidence: ${(fish.detection.confidence * 100).toFixed(1)}%</div>
                                        <div class="measurement-value">Measurement Type: ${fish.measurement_type.replace('_', ' ')}</div>
                                    </div>
                                    
                                    <div class="measurement-group">
                                        <div class="measurement-title">📏 Distance</div>
                                        <div class="measurement-value">Distance: ${fish.distance.distance_cm} cm</div>
                                        <div class="measurement-value">Distance: ${fish.distance.distance_inches}" inches</div>
                                        <div class="measurement-value">Method: ${fish.distance.estimation_method}</div>
                                    </div>
                                    
                                    <div class="measurement-group">
                                        <div class="measurement-title">📐 Dimensions</div>
                                        <div class="measurement-value">Length: ${fish.dimensions.length_cm.toFixed(1)} cm</div>
                                        <div class="measurement-value">Length: ${fish.dimensions.length_inches.toFixed(1)}" inches</div>
                                        <div class="measurement-value">Height: ${fish.dimensions.height_cm.toFixed(1)} cm</div>
                                    </div>
                                    
                                    <div class="measurement-group">
                                        <div class="measurement-title">⚖️ Weight Estimation</div>
                                        <div class="measurement-value"><strong>${fish.weight_calculation.weight_kg.toFixed(3)} kg</strong></div>
                                        <div class="measurement-value"><strong>${fish.weight_calculation.weight_pounds.toFixed(2)} pounds</strong></div>
                                        <div class="measurement-value">${fish.weight_calculation.weight_grams.toFixed(0)} grams</div>
                                        <div class="measurement-value">Estimated Girth: ${fish.weight_calculation.estimated_girth_cm.toFixed(1)} cm</div>
                                    </div>
                                </div>
                        `;
                        
                        // Species classification if available
                        if (fish.species_classification) {
                            html += `
                                <div class="measurement-group">
                                    <div class="measurement-title">🔬 Species Classification</div>
                                    <div class="measurement-value">Species: ${fish.species_classification.predicted_species}</div>
                                    <div class="measurement-value">Confidence: ${(fish.species_classification.confidence * 100).toFixed(1)}%</div>
                                    <div class="measurement-value">ID: ${fish.species_classification.species_id}</div>
                                </div>
                            `;
                        }
                        
                        html += '</div>';
                    });
                } else {
                    html += '<div class="error">No fish detected in the image.</div>';
                }
                
                // Processing time
                html += `
                    <div style="margin-top: 20px; padding: 10px; background: #ecf0f1; border-radius: 5px;">
                        <strong>⏱️ Processing Time:</strong> ${data.processing_time.total.toFixed(2)}s
                        (Detection: ${data.processing_time.detection.toFixed(2)}s)
                    </div>
                `;
                
                document.getElementById('results').innerHTML = html;
                document.getElementById('results').style.display = 'block';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/analyze-distance-weight', methods=['POST'])
def analyze_distance_weight():
    """Analyze uploaded image for fish detection, distance, and weight."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fish_analysis_{timestamp}_{file.filename}"
        filepath = os.path.join('uploads', filename)
        
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Analyze image
        results = analyzer.analyze_image_with_distance_weight(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info')
def api_info():
    """API information endpoint."""
    return jsonify({
        'name': 'Fish Distance & Weight Analyzer API',
        'version': '1.0.0',
        'description': 'Advanced fish detection with distance estimation and weight calculation',
        'features': [
            'YOLOv12 fish detection (28+ species)',
            'MediaPipe-based distance estimation',
            'Multiple weight calculation formulas',
            'Species-specific weight estimation',
            'Real-time dimensional analysis'
        ],
        'weight_formulas': {
            'girth_based': '(length × girth²) ÷ 800',
            'length_based': '(length³) ÷ 1200',
            'species_specific': 'Custom formulas per species'
        },
        'measurement_types': analyzer.measurement_types,
        'endpoints': {
            '/': 'Web interface',
            '/analyze-distance-weight': 'POST - Image analysis',
            '/api/info': 'GET - API information'
        }
    })

if __name__ == '__main__':
    print("🐟 Starting Advanced Fish Distance & Weight Analyzer...")
    print("=" * 60)
    
    # Check components
    if analyzer.detector:
        print("✅ YOLOv12 Fish Detector: Ready")
    else:
        print("❌ YOLOv12 Fish Detector: Failed")
    
    if analyzer.classifier:
        print("✅ Fish Classifier: Ready")
    else:
        print("❌ Fish Classifier: Failed")
    
    print("✅ MediaPipe Distance Estimation: Ready")
    print("✅ Weight Calculation Formulas: Ready")
    
    print(f"\n🚀 Starting server...")
    print(f"📡 Web Interface: http://localhost:5005")
    print(f"🔍 API Endpoint: POST http://localhost:5005/analyze-distance-weight")
    print(f"📊 API Info: GET http://localhost:5005/api/info")
    print(f"⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5005, debug=True) 
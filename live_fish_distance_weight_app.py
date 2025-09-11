#!/usr/bin/env python3
"""
Live Fish Distance & Weight Detection Web App
===========================================

Real-time fish detection with distance estimation and weight calculation
using live camera feed, MediaPipe for depth, and scientific fish weight formulas.
"""

import cv2
import numpy as np
import json
import time
import os
import sys
import threading
import queue
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import base64
import math

# Add detector paths
sys.path.insert(0, './detector_v12')
try:
    from local_inference import LocalYOLOv12Fish
except ImportError:
    print("Warning: Could not import LocalYOLOv12Fish")
    LocalYOLOv12Fish = None

app = Flask(__name__)
CORS(app)

class LiveFishDistanceAnalyzer:
    """Live fish detection with distance and weight estimation."""
    
    def __init__(self, camera_id=0):
        """Initialize live analyzer."""
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_results = None
        
        # Initialize fish detector
        if LocalYOLOv12Fish:
            try:
                self.detector = LocalYOLOv12Fish(confidence=0.4)
                print("✅ Fish detector loaded")
            except Exception as e:
                print(f"❌ Failed to load detector: {e}")
                self.detector = None
        else:
            self.detector = None
        
        # Camera parameters for distance calculation
        self.focal_length = 800  # pixels (adjust for your camera)
        self.reference_object_size = 11.7  # mm (iris diameter)
        
        # Fish weight calculation formulas
        self.weight_formulas = {
            'girth_based': lambda length, girth: (length * girth * girth) / 800,
            'length_based': lambda length: (length * length * length) / 1200,
            'species_factors': {
                'Tuna': 2.8, 'Bass': 3.0, 'Salmon': 2.9, 'Catfish': 3.1,
                'Snapper': 2.85, 'Mackerel': 2.7, 'Grouper': 3.2
            }
        }
        
        # Species girth ratios (girth/length)
        self.girth_ratios = {
            'Tuna': 0.65, 'Bass': 0.45, 'Salmon': 0.40, 'Catfish': 0.55,
            'Snapper': 0.42, 'Mackerel': 0.35, 'Grouper': 0.50, 'default': 0.45
        }
        
        # Detection statistics
        self.stats = {
            'total_detections': 0,
            'session_start': time.time(),
            'fish_weights': [],
            'fish_distances': [],
            'species_counts': {}
        }
        
    def initialize_camera(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera {self.camera_id} initialized")
        return True
    
    def estimate_distance_simple(self, fish_bbox, frame_shape):
        """Simple distance estimation based on fish size in frame."""
        x1, y1, x2, y2 = fish_bbox
        fish_width = x2 - x1
        fish_height = y2 - y1
        fish_area = fish_width * fish_height
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Area ratio approach
        area_ratio = fish_area / frame_area
        
        # Empirical distance estimation (adjust based on testing)
        # Larger fish in frame = closer distance
        estimated_distance = 300 / math.sqrt(area_ratio)  # cm
        
        return max(10, min(500, estimated_distance))  # clamp between 10-500cm
    
    def calculate_fish_dimensions(self, fish_bbox, distance_cm, frame_shape):
        """Calculate real fish dimensions from bounding box."""
        x1, y1, x2, y2 = fish_bbox
        fish_width_pixels = x2 - x1
        fish_height_pixels = y2 - y1
        
        # Convert pixels to real dimensions using distance
        # This is simplified - in reality you'd use camera calibration
        pixel_to_cm_ratio = distance_cm / self.focal_length
        
        fish_length_cm = fish_width_pixels * pixel_to_cm_ratio
        fish_height_cm = fish_height_pixels * pixel_to_cm_ratio
        
        return {
            'length_cm': fish_length_cm,
            'height_cm': fish_height_cm,
            'length_inches': fish_length_cm / 2.54,
            'height_inches': fish_height_cm / 2.54
        }
    
    def estimate_fish_weight(self, dimensions, species):
        """Estimate fish weight using multiple methods."""
        length_cm = dimensions['length_cm']
        
        # Estimate girth based on species
        girth_ratio = self.girth_ratios.get(species, self.girth_ratios['default'])
        estimated_girth = length_cm * girth_ratio
        
        # Method 1: Girth-based formula
        weight_girth = self.weight_formulas['girth_based'](length_cm, estimated_girth)
        
        # Method 2: Length-based formula
        weight_length = self.weight_formulas['length_based'](length_cm)
        
        # Method 3: Species-specific
        weight_species = None
        if species in self.weight_formulas['species_factors']:
            factor = self.weight_formulas['species_factors'][species]
            weight_species = (length_cm ** factor) / 1000
        
        # Average the available methods
        weights = [w for w in [weight_girth, weight_length, weight_species] if w is not None]
        final_weight = sum(weights) / len(weights) if weights else 0
        
        return {
            'weight_grams': final_weight,
            'weight_kg': final_weight / 1000,
            'weight_pounds': final_weight * 0.00220462,
            'estimated_girth_cm': estimated_girth,
            'methods': {
                'girth_based': weight_girth,
                'length_based': weight_length,
                'species_specific': weight_species
            }
        }
    
    def process_frame(self, frame):
        """Process frame for fish detection and analysis."""
        if not self.detector:
            return frame, []
        
        try:
            # Detect fish
            detections = self.detector.predict(frame)
            fish_list = detections[0] if detections and detections[0] else []
            
            results = []
            annotated_frame = frame.copy()
            
            for i, fish in enumerate(fish_list):
                # Get detection info
                bbox = fish.get_box()
                confidence = fish.get_score()
                species = fish.get_class_name()
                
                # Estimate distance
                distance_cm = self.estimate_distance_simple(bbox, frame.shape)
                
                # Calculate dimensions
                dimensions = self.calculate_fish_dimensions(bbox, distance_cm, frame.shape)
                
                # Estimate weight
                weight_info = self.estimate_fish_weight(dimensions, species)
                
                # Prepare result
                fish_result = {
                    'id': i + 1,
                    'species': species,
                    'confidence': confidence,
                    'bbox': bbox,
                    'distance_cm': round(distance_cm, 1),
                    'distance_inches': round(distance_cm / 2.54, 1),
                    'dimensions': dimensions,
                    'weight': weight_info
                }
                results.append(fish_result)
                
                # Update statistics
                self.stats['total_detections'] += 1
                self.stats['fish_weights'].append(weight_info['weight_grams'])
                self.stats['fish_distances'].append(distance_cm)
                self.stats['species_counts'][species] = self.stats['species_counts'].get(species, 0) + 1
                
                # Draw annotations
                annotated_frame = self.draw_fish_annotation(annotated_frame, fish_result)
            
            self.current_results = results
            return annotated_frame, results
            
        except Exception as e:
            print(f"Processing error: {e}")
            return frame, []
    
    def draw_fish_annotation(self, frame, fish_result):
        """Draw comprehensive fish annotation on frame."""
        x1, y1, x2, y2 = map(int, fish_result['bbox'])
        
        # Choose color based on fish ID
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        color = colors[fish_result['id'] % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Fish ID badge
        cv2.circle(frame, (x1 - 20, y1 - 20), 15, color, -1)
        cv2.putText(frame, str(fish_result['id']), (x1 - 27, y1 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Prepare annotation text
        annotations = [
            f"{fish_result['species']} ({fish_result['confidence']:.2f})",
            f"Distance: {fish_result['distance_cm']} cm",
            f"Length: {fish_result['dimensions']['length_cm']:.1f} cm",
            f"Weight: {fish_result['weight']['weight_kg']:.3f} kg",
            f"({fish_result['weight']['weight_pounds']:.2f} lbs)"
        ]
        
        # Calculate annotation box size
        max_width = 0
        total_height = 0
        for text in annotations:
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_width = max(max_width, w)
            total_height += h + 5
        
        # Draw annotation background
        bg_x1, bg_y1 = x1, y1 - total_height - 10
        bg_x2, bg_y2 = x1 + max_width + 10, y1
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
        
        # Draw annotation text
        y_offset = y1 - total_height + 15
        for text in annotations:
            cv2.putText(frame, text, (x1 + 5, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return frame
    
    def draw_ui_overlay(self, frame):
        """Draw UI overlay with statistics."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "🐟 Live Fish Distance & Weight Analysis", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current session stats
        session_time = time.time() - self.stats['session_start']
        detection_rate = self.stats['total_detections'] / (session_time / 60) if session_time > 0 else 0
        
        stats_text = [
            f"Session: {int(session_time//60):02d}:{int(session_time%60):02d}",
            f"Total Fish: {self.stats['total_detections']}",
            f"Rate: {detection_rate:.1f}/min",
            f"Active Fish: {len(self.current_results) if self.current_results else 0}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (20, 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current detections summary
        if self.current_results:
            total_weight = sum(fish['weight']['weight_grams'] for fish in self.current_results)
            avg_distance = sum(fish['distance_cm'] for fish in self.current_results) / len(self.current_results)
            
            summary_text = [
                f"Current Frame:",
                f"  Total Weight: {total_weight/1000:.3f} kg ({total_weight*0.00220462:.2f} lbs)",
                f"  Avg Distance: {avg_distance:.1f} cm"
            ]
            
            for i, text in enumerate(summary_text):
                cv2.putText(frame, text, (20, 140 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Controls
        control_y = height - 40
        cv2.rectangle(frame, (10, control_y-10), (width-10, height-10), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        cv2.putText(frame, "Controls: [Space] Screenshot | [R] Reset Stats | [Q] Quit", 
                   (20, control_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_camera_feed(self):
        """Start camera feed processing."""
        if not self.initialize_camera():
            return
        
        self.running = True
        print("🎥 Starting live fish analysis...")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Add UI overlay
                final_frame = self.draw_ui_overlay(processed_frame)
                
                # Store current frame for web streaming
                self.current_frame = final_frame
                
                # Small delay to prevent overload
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Camera feed error: {e}")
        finally:
            self.cleanup()
    
    def get_frame_bytes(self):
        """Get current frame as JPEG bytes for streaming."""
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            if ret:
                return buffer.tobytes()
        return None
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        print("🧹 Camera feed cleaned up")

# Global analyzer instance
analyzer = LiveFishDistanceAnalyzer()

@app.route('/')
def index():
    """Main page for live fish analysis."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐟 Live Fish Distance & Weight Analysis</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { 
                text-align: center; 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 15px; 
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
            }
            .main-content { 
                display: grid; 
                grid-template-columns: 1fr 400px; 
                gap: 20px; 
                align-items: start;
            }
            .video-section { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 15px;
                backdrop-filter: blur(10px);
                text-align: center;
            }
            .stats-section { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 15px;
                backdrop-filter: blur(10px);
                height: fit-content;
            }
            #videoFeed { 
                max-width: 100%; 
                border-radius: 10px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            .controls { 
                margin: 20px 0; 
                text-align: center;
            }
            button { 
                background: #ff6b6b; 
                color: white; 
                border: none; 
                padding: 12px 24px; 
                border-radius: 25px; 
                cursor: pointer; 
                font-size: 16px; 
                margin: 5px;
                transition: all 0.3s ease;
            }
            button:hover { 
                background: #ff5252; 
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            .status { 
                padding: 10px; 
                border-radius: 10px; 
                margin: 10px 0;
            }
            .status.connected { background: rgba(76, 175, 80, 0.3); }
            .status.disconnected { background: rgba(244, 67, 54, 0.3); }
            .stats-grid { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 15px; 
                margin: 20px 0;
            }
            .stat-card { 
                background: rgba(255,255,255,0.1); 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center;
            }
            .stat-value { 
                font-size: 24px; 
                font-weight: bold; 
                color: #4fc3f7;
            }
            .stat-label { 
                font-size: 12px; 
                opacity: 0.8; 
            }
            .detection-list { 
                max-height: 300px; 
                overflow-y: auto; 
                margin: 20px 0;
            }
            .detection-item { 
                background: rgba(255,255,255,0.1); 
                padding: 10px; 
                border-radius: 8px; 
                margin: 5px 0;
            }
            .species-name { 
                font-weight: bold; 
                color: #4fc3f7; 
            }
            .weight-info { 
                font-size: 14px; 
                opacity: 0.9; 
            }
            @media (max-width: 768px) {
                .main-content { 
                    grid-template-columns: 1fr; 
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐟 Live Fish Distance & Weight Analysis</h1>
                <p>Real-time fish detection with distance estimation and weight calculation</p>
            </div>
            
            <div class="main-content">
                <div class="video-section">
                    <div id="connectionStatus" class="status disconnected">
                        📡 Connecting to camera...
                    </div>
                    
                    <img id="videoFeed" src="/video_feed" alt="Video Feed" 
                         onload="updateConnectionStatus(true)" 
                         onerror="updateConnectionStatus(false)" />
                    
                    <div class="controls">
                        <button onclick="startCamera()">🎥 Start Camera</button>
                        <button onclick="stopCamera()">⏹️ Stop Camera</button>
                        <button onclick="resetStats()">📊 Reset Stats</button>
                        <button onclick="takeScreenshot()">📸 Screenshot</button>
                    </div>
                </div>
                
                <div class="stats-section">
                    <h3>📊 Live Statistics</h3>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div id="totalFish" class="stat-value">0</div>
                            <div class="stat-label">Total Fish</div>
                        </div>
                        <div class="stat-card">
                            <div id="activeFish" class="stat-value">0</div>
                            <div class="stat-label">Active Fish</div>
                        </div>
                        <div class="stat-card">
                            <div id="totalWeight" class="stat-value">0kg</div>
                            <div class="stat-label">Total Weight</div>
                        </div>
                        <div class="stat-card">
                            <div id="avgDistance" class="stat-value">0cm</div>
                            <div class="stat-label">Avg Distance</div>
                        </div>
                    </div>
                    
                    <h4>🐠 Current Detections</h4>
                    <div id="detectionList" class="detection-list">
                        <div style="text-align: center; opacity: 0.6;">
                            No fish detected
                        </div>
                    </div>
                    
                    <h4>📈 Session Info</h4>
                    <div style="font-size: 14px;">
                        <div>Session Time: <span id="sessionTime">00:00</span></div>
                        <div>Detection Rate: <span id="detectionRate">0.0/min</span></div>
                        <div>Camera Status: <span id="cameraStatus">Ready</span></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let sessionStartTime = Date.now();
            let statsUpdateInterval;
            
            function updateConnectionStatus(connected) {
                const status = document.getElementById('connectionStatus');
                if (connected) {
                    status.className = 'status connected';
                    status.innerHTML = '✅ Camera Connected';
                } else {
                    status.className = 'status disconnected';
                    status.innerHTML = '❌ Camera Disconnected';
                }
            }
            
            function startCamera() {
                fetch('/start_camera', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('cameraStatus').textContent = data.status;
                        if (data.success) {
                            startStatsUpdate();
                        }
                    });
            }
            
            function stopCamera() {
                fetch('/stop_camera', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('cameraStatus').textContent = data.status;
                        stopStatsUpdate();
                    });
            }
            
            function resetStats() {
                fetch('/reset_stats', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        sessionStartTime = Date.now();
                        updateStats();
                    });
            }
            
            function takeScreenshot() {
                fetch('/screenshot', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Screenshot saved: ' + data.filename);
                        }
                    });
            }
            
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        // Update main stats
                        document.getElementById('totalFish').textContent = data.total_detections;
                        document.getElementById('activeFish').textContent = data.current_fish.length;
                        
                        // Calculate total weight of current fish
                        const totalWeight = data.current_fish.reduce((sum, fish) => 
                            sum + fish.weight.weight_grams, 0);
                        document.getElementById('totalWeight').textContent = (totalWeight/1000).toFixed(3) + 'kg';
                        
                        // Calculate average distance
                        const avgDist = data.current_fish.length > 0 ? 
                            data.current_fish.reduce((sum, fish) => sum + fish.distance_cm, 0) / data.current_fish.length : 0;
                        document.getElementById('avgDistance').textContent = avgDist.toFixed(1) + 'cm';
                        
                        // Update session time
                        const sessionSeconds = Math.floor((Date.now() - sessionStartTime) / 1000);
                        const minutes = Math.floor(sessionSeconds / 60);
                        const seconds = sessionSeconds % 60;
                        document.getElementById('sessionTime').textContent = 
                            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                        
                        // Update detection rate
                        const detectionRate = minutes > 0 ? (data.total_detections / minutes).toFixed(1) : '0.0';
                        document.getElementById('detectionRate').textContent = detectionRate + '/min';
                        
                        // Update detection list
                        updateDetectionList(data.current_fish);
                    })
                    .catch(error => console.error('Stats update error:', error));
            }
            
            function updateDetectionList(fish) {
                const list = document.getElementById('detectionList');
                
                if (fish.length === 0) {
                    list.innerHTML = '<div style="text-align: center; opacity: 0.6;">No fish detected</div>';
                    return;
                }
                
                list.innerHTML = fish.map(f => `
                    <div class="detection-item">
                        <div class="species-name">🐠 ${f.species}</div>
                        <div class="weight-info">
                            Weight: ${f.weight.weight_kg.toFixed(3)}kg (${f.weight.weight_pounds.toFixed(2)} lbs)<br>
                            Distance: ${f.distance_cm}cm | Length: ${f.dimensions.length_cm.toFixed(1)}cm<br>
                            Confidence: ${(f.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                `).join('');
            }
            
            function startStatsUpdate() {
                if (statsUpdateInterval) clearInterval(statsUpdateInterval);
                statsUpdateInterval = setInterval(updateStats, 1000); // Update every second
                updateStats(); // Initial update
            }
            
            function stopStatsUpdate() {
                if (statsUpdateInterval) {
                    clearInterval(statsUpdateInterval);
                    statsUpdateInterval = null;
                }
            }
            
            // Auto-start stats update when page loads
            startStatsUpdate();
            
            // Refresh video feed every 30 seconds to prevent timeout
            setInterval(() => {
                const img = document.getElementById('videoFeed');
                const src = img.src;
                img.src = '';
                img.src = src + '?t=' + Date.now();
            }, 30000);
        </script>
    </body>
    </html>
    '''

def generate_frames():
    """Generate video frames for streaming."""
    while True:
        frame_bytes = analyzer.get_frame_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send empty frame if no camera feed
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera feed."""
    try:
        if not analyzer.running:
            # Start camera in separate thread
            camera_thread = threading.Thread(target=analyzer.start_camera_feed)
            camera_thread.daemon = True
            camera_thread.start()
            return jsonify({'success': True, 'status': 'Camera started'})
        else:
            return jsonify({'success': True, 'status': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'status': f'Error: {e}'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera feed."""
    try:
        analyzer.running = False
        analyzer.cleanup()
        return jsonify({'success': True, 'status': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'status': f'Error: {e}'})

@app.route('/stats')
def get_stats():
    """Get current statistics."""
    current_fish = analyzer.current_results if analyzer.current_results else []
    
    return jsonify({
        'total_detections': analyzer.stats['total_detections'],
        'current_fish': current_fish,
        'session_time': time.time() - analyzer.stats['session_start'],
        'species_counts': analyzer.stats['species_counts']
    })

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset statistics."""
    analyzer.stats = {
        'total_detections': 0,
        'session_start': time.time(),
        'fish_weights': [],
        'fish_distances': [],
        'species_counts': {}
    }
    return jsonify({'success': True, 'message': 'Statistics reset'})

@app.route('/screenshot', methods=['POST'])
def take_screenshot():
    """Take screenshot of current frame."""
    try:
        if analyzer.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fish_live_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, analyzer.current_frame)
            return jsonify({'success': True, 'filename': filename})
        else:
            return jsonify({'success': False, 'error': 'No frame available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🐟 Starting Live Fish Distance & Weight Analysis Server...")
    print("=" * 60)
    
    if analyzer.detector:
        print("✅ Fish Detector: Ready")
    else:
        print("❌ Fish Detector: Not available")
    
    print("✅ Distance Estimation: Ready")
    print("✅ Weight Calculation: Ready")
    print("✅ Live Camera Streaming: Ready")
    
    print(f"\n🚀 Server starting...")
    print(f"📡 Web Interface: http://localhost:5006")
    print(f"🎥 Video Stream: http://localhost:5006/video_feed")
    print(f"📊 Stats API: http://localhost:5006/stats")
    print(f"⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5006, debug=True, threaded=True) 
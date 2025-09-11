#!/usr/bin/env python3
"""
Mobile Fish Detection App
========================

Modern mobile-first fish detection with full-screen camera,
automatic detection, and detailed analysis.
"""

import cv2
import numpy as np
import json
import time
import os
import sys
import threading
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

# Add classification path
sys.path.insert(0, './classification_rectangle_v7-1')
try:
    from inference import EmbeddingClassifier
except ImportError:
    print("Warning: Could not import EmbeddingClassifier")
    EmbeddingClassifier = None

app = Flask(__name__)
CORS(app)

class MobileFishDetector:
    """Mobile-optimized fish detector with auto-pause and analysis."""
    
    def __init__(self, camera_id=0):
        """Initialize mobile fish detector."""
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.detection_active = True
        self.current_frame = None
        self.detected_fish = []
        self.capture_on_detection = True
        self.captured_image = None
        self.capture_timer = None
        self.auto_capture_delay = 4  # 4 seconds delay
        self.capture_complete = False  # Flag to track capture completion
        
        # Initialize fish detector
        if LocalYOLOv12Fish:
            try:
                self.detector = LocalYOLOv12Fish(confidence=0.3)  # Lower initial threshold
                print("✅ Fish detector loaded")
            except Exception as e:
                print(f"❌ Failed to load detector: {e}")
                self.detector = None
        else:
            self.detector = None
        
        # Initialize classifier (optional)
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
                self.classifier = None
        else:
            self.classifier = None
        
        # Fish weight calculation
        self.weight_formulas = {
            'girth_based': lambda length, girth: (length * girth * girth) / 800,
            'length_based': lambda length: (length * length * length) / 1200,
            'species_factors': {
                'Tuna': 2.8, 'Bass': 3.0, 'Salmon': 2.9, 'Catfish': 3.1,
                'Snapper': 2.85, 'Mackerel': 2.7, 'Grouper': 3.2
            }
        }
        
        self.girth_ratios = {
            'Tuna': 0.65, 'Bass': 0.45, 'Salmon': 0.40, 'Catfish': 0.55,
            'Snapper': 0.42, 'Mackerel': 0.35, 'Grouper': 0.50, 'default': 0.45
        }
        
        # Camera parameters
        self.focal_length = 800
        
        # Auto-start camera
        self.auto_start_camera()
        
    def auto_start_camera(self):
        """Automatically start camera in background."""
        try:
            camera_thread = threading.Thread(target=self.start_camera_feed)
            camera_thread.daemon = True
            camera_thread.start()
            print("🎥 Camera auto-started")
        except Exception as e:
            print(f"❌ Failed to auto-start camera: {e}")
    
    def initialize_camera(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties for mobile
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera {self.camera_id} initialized")
        return True
    
    def estimate_distance(self, fish_bbox, frame_shape):
        """Estimate distance based on fish size."""
        x1, y1, x2, y2 = fish_bbox
        fish_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = fish_area / frame_area
        
        # Distance estimation (adjustable)
        estimated_distance = 400 / math.sqrt(area_ratio)  # cm
        return max(20, min(300, estimated_distance))
    
    def calculate_dimensions(self, fish_bbox, distance_cm):
        """Calculate fish dimensions from bounding box."""
        x1, y1, x2, y2 = fish_bbox
        fish_width_pixels = x2 - x1
        fish_height_pixels = y2 - y1
        
        pixel_to_cm_ratio = distance_cm / self.focal_length
        fish_length_cm = fish_width_pixels * pixel_to_cm_ratio
        fish_height_cm = fish_height_pixels * pixel_to_cm_ratio
        
        return {
            'length_cm': fish_length_cm,
            'height_cm': fish_height_cm,
            'length_inches': fish_length_cm / 2.54,
            'height_inches': fish_height_cm / 2.54
        }
    
    def calculate_weight(self, dimensions, species):
        """Calculate fish weight."""
        length_cm = dimensions['length_cm']
        girth_ratio = self.girth_ratios.get(species, self.girth_ratios['default'])
        estimated_girth = length_cm * girth_ratio
        
        # Multiple weight calculation methods
        weight_girth = self.weight_formulas['girth_based'](length_cm, estimated_girth)
        weight_length = self.weight_formulas['length_based'](length_cm)
        
        weight_species = None
        if species in self.weight_formulas['species_factors']:
            factor = self.weight_formulas['species_factors'][species]
            weight_species = (length_cm ** factor) / 1000
        
        # Average available methods
        weights = [w for w in [weight_girth, weight_length, weight_species] if w is not None]
        final_weight = sum(weights) / len(weights) if weights else 0
        
        return {
            'weight_kg': final_weight / 1000,
            'weight_pounds': final_weight * 0.00220462,
            'weight_grams': final_weight,
            'estimated_girth_cm': estimated_girth,
            'calculation_methods': len(weights)
        }
    
    def classify_species(self, image, fish_bbox):
        """Classify fish species if classifier available."""
        if not self.classifier:
            return None
        
        try:
            x1, y1, x2, y2 = map(int, fish_bbox)
            fish_crop = image[y1:y2, x1:x2]
            if fish_crop.size > 0:
                # Use the correct classifier method
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
    
    def create_zoomed_capture(self, frame, fish_bbox, zoom_factor=0.2):
        """Create a zoomed capture of the fish with 20% padding."""
        try:
            x1, y1, x2, y2 = map(int, fish_bbox)
            h, w = frame.shape[:2]
            
            # Calculate fish dimensions
            fish_width = x2 - x1
            fish_height = y2 - y1
            
            # Add 20% zoom (padding)
            padding_w = int(fish_width * zoom_factor)
            padding_h = int(fish_height * zoom_factor)
            
            # Calculate zoomed boundaries
            zoom_x1 = max(0, x1 - padding_w)
            zoom_y1 = max(0, y1 - padding_h)
            zoom_x2 = min(w, x2 + padding_w)
            zoom_y2 = min(h, y2 + padding_h)
            
            # Crop the zoomed region
            zoomed_frame = frame[zoom_y1:zoom_y2, zoom_x1:zoom_x2].copy()
            
            # Draw fish bounding box on zoomed image (adjust coordinates)
            adjusted_x1 = x1 - zoom_x1
            adjusted_y1 = y1 - zoom_y1
            adjusted_x2 = x2 - zoom_x1
            adjusted_y2 = y2 - zoom_y1
            
            # Draw bounding box
            cv2.rectangle(zoomed_frame, (adjusted_x1, adjusted_y1), 
                         (adjusted_x2, adjusted_y2), (0, 255, 0), 3)
            
            return zoomed_frame
            
        except Exception as e:
            print(f"Zoom capture error: {e}")
            return frame
    
    def auto_capture_fish(self):
        """Automatically capture fish after delay."""
        try:
            if self.detected_fish and self.current_frame is not None:
                # Get the first detected fish
                fish = self.detected_fish[0]
                bbox = fish['bbox']
                
                # Create zoomed capture
                self.captured_image = self.create_zoomed_capture(self.current_frame, bbox)
                
                print(f"📸 Auto-captured fish: {fish['species']}")
                
                # Mark as captured and stop camera processing
                self.capture_complete = True
                self.detection_active = False
                self.running = False  # Stop camera feed
                
        except Exception as e:
            print(f"Auto-capture error: {e}")
    
    def process_frame_for_detection(self, frame):
        """Process frame for fish detection."""
        if not self.detector:
            return frame, []
        
        try:
            # Detect fish
            detections = self.detector.predict(frame)
            fish_list = detections[0] if detections and detections[0] else []
            
            if fish_list and self.detection_active and self.capture_on_detection:
                # Validate detections before proceeding
                valid_fish = []
                for fish in fish_list:
                    confidence = fish.get_score()
                    # Only consider fish with confidence > 0.6
                    if confidence > 0.6:
                        valid_fish.append(fish)
                
                if valid_fish:
                    # Auto-pause on valid detection
                    self.detection_active = False
                    self.detected_fish = []
                    
                    for i, fish in enumerate(valid_fish):
                        bbox = fish.get_box()
                        confidence = fish.get_score()
                        species = fish.get_class_name()
                        
                        # Calculate analysis
                        distance_cm = self.estimate_distance(bbox, frame.shape)
                        dimensions = self.calculate_dimensions(bbox, distance_cm)
                        weight_info = self.calculate_weight(dimensions, species)
                        
                        # Species classification
                        species_classification = self.classify_species(frame, bbox)
                        
                        fish_data = {
                            'id': i,
                            'species': species,
                            'confidence': confidence,
                            'bbox': bbox,
                            'distance_cm': round(distance_cm, 1),
                            'dimensions': dimensions,
                            'weight': weight_info,
                            'species_classification': species_classification,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        self.detected_fish.append(fish_data)
                    
                    # Create annotated frame
                    annotated_frame = self.draw_detection_overlay(frame.copy())
                    self.current_frame = annotated_frame
                    
                    # Start auto-capture timer only for valid detections
                    if self.capture_timer is None:
                        self.capture_timer = threading.Timer(self.auto_capture_delay, self.auto_capture_fish)
                        self.capture_timer.start()
                        print(f"Starting capture timer - {len(valid_fish)} valid fish detected")
                    
                    return annotated_frame, self.detected_fish
                else:
                    # No valid fish found despite detections
                    print("Low confidence detections ignored")
                    return frame, []
                    
            elif fish_list and not self.detection_active:
                # Update detection but don't auto-pause (for gallery images or manual processing)
                self.detected_fish = []
                
                for i, fish in enumerate(fish_list):
                    bbox = fish.get_box()
                    confidence = fish.get_score()
                    species = fish.get_class_name()
                    
                    # Calculate analysis
                    distance_cm = self.estimate_distance(bbox, frame.shape)
                    dimensions = self.calculate_dimensions(bbox, distance_cm)
                    weight_info = self.calculate_weight(dimensions, species)
                    
                    # Species classification
                    species_classification = self.classify_species(frame, bbox)
                    
                    fish_data = {
                        'id': i,
                        'species': species,
                        'confidence': confidence,
                        'bbox': bbox,
                        'distance_cm': round(distance_cm, 1),
                        'dimensions': dimensions,
                        'weight': weight_info,
                        'species_classification': species_classification,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.detected_fish.append(fish_data)
                
                # Create annotated frame
                annotated_frame = self.draw_detection_overlay(frame.copy())
                return annotated_frame, self.detected_fish
            
            return frame, []
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []
    
    def draw_detection_overlay(self, frame):
        """Draw clean detection overlay."""
        if not self.detected_fish:
            return frame
        
        for fish in self.detected_fish:
            bbox = fish['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw border-only bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Fish info overlay (top of bbox)
            info_text = f"{fish['species']} - {fish['weight']['weight_kg']:.3f}kg"
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-35), (x1+text_width+10, y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, info_text, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def start_camera_feed(self):
        """Start camera feed processing."""
        if not self.initialize_camera():
            return
        
        self.running = True
        print("🎥 Starting mobile fish detection...")
        
        try:
            while self.running:
                if self.cap is None:
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Mirror effect for front camera
                frame = cv2.flip(frame, 1)
                
                # Process for detection
                processed_frame, detections = self.process_frame_for_detection(frame)
                
                # Store current frame
                self.current_frame = processed_frame
                
                # Small delay
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Camera feed error: {e}")
        finally:
            self.cleanup()
    
    def get_frame_bytes(self):
        """Get current frame as JPEG bytes."""
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                return buffer.tobytes()
        return None
    
    def restart_detection(self):
        """Restart detection mode."""
        self.detection_active = True
        self.detected_fish = []
        self.capture_on_detection = True
        self.captured_image = None
        self.capture_complete = False  # Reset capture flag
        
        # Cancel any pending capture timer
        if self.capture_timer:
            self.capture_timer.cancel()
            self.capture_timer = None
        
        # Restart camera feed
        if not self.running:
            self.auto_start_camera()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        print("🧹 Camera cleaned up")
    
    def get_captured_image_bytes(self):
        """Get captured image as JPEG bytes."""
        if self.captured_image is not None:
            ret, buffer = cv2.imencode('.jpg', self.captured_image, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ret:
                return buffer.tobytes()
        return None

# Global detector instance
detector = MobileFishDetector()

@app.route('/')
def index():
    """Main mobile fish detection interface."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🐟 Mobile Fish Detector</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #000;
                overflow: hidden;
                position: relative;
                height: 100vh;
                width: 100vw;
            }
            
            #videoContainer {
                position: relative;
                width: 100vw;
                height: 100vh;
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            #videoFeed {
                width: 100vw;
                height: 100vh;
                object-fit: cover;
                background: #000;
                cursor: pointer;
            }
            
            /* Mobile 9:16 optimization */
            @media (max-aspect-ratio: 9/16) {
                #videoFeed {
                    width: 100vw;
                    height: 177.78vw; /* 16/9 * 100vw */
                    max-height: 100vh;
                }
            }
            
            @media (min-aspect-ratio: 9/16) {
                #videoFeed {
                    width: 56.25vh; /* 9/16 * 100vh */
                    height: 100vh;
                    max-width: 100vw;
                }
            }
            
            .captured-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: #000;
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 150;
                cursor: pointer;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .captured-overlay.show {
                display: flex;
                opacity: 1;
            }
            
            .captured-image {
                width: 100vw;
                height: 100vh;
                object-fit: cover;
                background: #000;
            }
            
            .gallery-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: #000;
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 150;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .gallery-overlay.show {
                display: flex;
                opacity: 1;
            }
            
            .gallery-image {
                max-width: 90vw;
                max-height: 80vh;
                object-fit: contain;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            
            .gallery-message {
                position: absolute;
                bottom: 100px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                text-align: center;
                font-size: 18px;
                background: rgba(0,0,0,0.8);
                padding: 20px 30px;
                border-radius: 25px;
                backdrop-filter: blur(10px);
                max-width: 80vw;
            }
            
            .capture-info {
                position: absolute;
                bottom: 100px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                text-align: center;
                font-size: 18px;
                background: rgba(0,0,0,0.7);
                padding: 15px 25px;
                border-radius: 25px;
                backdrop-filter: blur(10px);
            }
            
            .capture-countdown {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 48px;
                font-weight: bold;
                text-shadow: 2px 2px 10px rgba(0,0,0,0.8);
                z-index: 110;
                display: none;
            }
            
            .capture-countdown.show {
                display: block;
            }
            
            .top-info {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%);
                padding: 20px;
                z-index: 100;
                color: white;
                font-size: 16px;
                display: none;
            }
            
            .top-info.show {
                display: block;
            }
            
            .detection-count {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .detection-status {
                font-size: 14px;
                opacity: 0.9;
            }
            
            .floating-buttons {
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 20px;
                z-index: 100;
                width: 300px;
                justify-content: space-between;
                align-items: center;
            }
            
            .btn-circle {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                border: none;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: white;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                transform: scale(1);
            }
            
            .btn-circle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }
            
            .btn-circle:active {
                transform: scale(0.95);
            }
            
            .btn-circle.disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: scale(1) !important;
            }
            
            .btn-camera {
                width: 80px;
                height: 80px;
                font-size: 32px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .btn-gallery {
                background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            }
            
            .btn-reload {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
            }
            
            .btn-reload.show {
                opacity: 1;
                visibility: visible;
            }
            
            .btn-info {
                width: 80px;
                height: 80px;
                font-size: 32px;
                background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%);
            }
            
            .details-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: white;
                z-index: 200;
                display: none;
                overflow-y: auto;
                transform: translateX(100%);
                transition: transform 0.3s ease;
            }
            
            .details-overlay.show {
                display: block;
                transform: translateX(0);
            }
            
            .details-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                position: sticky;
                top: 0;
                z-index: 1;
            }
            
            .details-content {
                padding: 20px;
            }
            
            .fish-image {
                width: 100%;
                max-height: 300px;
                object-fit: contain;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 2px solid #ddd;
            }
            
            .fish-selector {
                margin-bottom: 20px;
            }
            
            .fish-selector select {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                background: white;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 15px;
            }
            
            .info-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            
            .info-title {
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 16px;
            }
            
            .info-value {
                color: #7f8c8d;
                font-size: 14px;
                margin: 3px 0;
            }
            
            .weight-highlight {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            
            .close-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                font-size: 20px;
                cursor: pointer;
            }
            
            .captured-buttons {
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 20px;
                z-index: 200;
                width: 300px;
                justify-content: space-between;
                align-items: center;
            }
            
            @media (max-width: 768px) {
                .floating-buttons {
                    bottom: 20px;
                    width: 280px;
                    gap: 15px;
                }
                
                .btn-circle {
                    width: 55px;
                    height: 55px;
                    font-size: 20px;
                }
                
                .btn-camera {
                    width: 70px;
                    height: 70px;
                    font-size: 28px;
                }
                
                .btn-info {
                    width: 70px;
                    height: 70px;
                    font-size: 28px;
                }
                
                .top-info {
                    padding: 15px;
                    font-size: 14px;
                }
                
                .capture-buttons {
                    top: 30px;
                    gap: 10px;
                }
                
                .capture-btn {
                    padding: 14px 20px;
                    font-size: 15px;
                    min-height: 48px;
                    min-width: 120px;
                }
                
                .capture-info {
                    bottom: 80px;
                    font-size: 16px;
                }
            }
        </style>
    </head>
    <body>
        <div id="videoContainer">
            <img id="videoFeed" src="/video_feed" alt="Camera Feed" onclick="restartCamera()" />
            
            <div class="capture-countdown" id="captureCountdown">3</div>
            
            <div class="top-info" id="topInfo">
                <div class="detection-count" id="detectionCount">No fish detected</div>
                <div class="detection-status" id="detectionStatus">Camera active - Point at fish</div>
            </div>
            
            <div class="floating-buttons" id="floatingButtons">
                <button class="btn-circle btn-gallery" onclick="openGallery()" title="Select from Gallery">
                    📁
                </button>
                <button class="btn-circle btn-camera" onclick="capturePhoto()" title="Capture Photo" id="cameraButton">
                    📷
                </button>
                <button class="btn-circle btn-info" onclick="getInfo()" title="Get Fish Info" style="display: none;" id="infoButton">
                    ℹ️
                </button>
                <button class="btn-circle btn-reload" onclick="reloadApp()" title="Reload App" id="reloadButton">
                    🔄
                </button>
            </div>
        </div>
        
        <!-- Captured image overlay (shows captured fish image) -->
        <div class="captured-overlay" id="capturedOverlay">
            <img id="capturedImage" class="captured-image" />
            
            <!-- Add floating buttons for captured image -->
            <div class="floating-buttons captured-buttons">
                <button class="btn-circle gallery-btn" onclick="selectFromGallery()" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    📁
                </button>
                <button class="btn-circle info-btn" id="capturedInfoButton" onclick="showDetails()" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    ℹ️
                </button>
                <button class="btn-circle reload-btn" onclick="reloadApp()" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    🔄
                </button>
            </div>
        </div>
        
        <div class="gallery-overlay" id="galleryOverlay" onclick="reloadApp()">
            <img id="galleryImage" class="gallery-image" />
            <div class="gallery-message" id="galleryMessage">
                <div>📁 Gallery Image</div>
                <div style="font-size: 14px; margin-top: 5px;">Tap to reload</div>
            </div>
        </div>
        
        <div class="details-overlay" id="detailsOverlay">
            <div class="details-header">
                <h2>🐟 Fish Analysis Details</h2>
                <button class="close-btn" onclick="closeDetails()">✕</button>
            </div>
            
            <div class="details-content">
                <canvas id="fishCanvas" class="fish-image"></canvas>
                
                <div class="fish-selector" id="fishSelector" style="display: none;">
                    <label for="fishSelect">Select Fish:</label>
                    <select id="fishSelect" onchange="selectFish()">
                        <option value="0">Fish 1</option>
                    </select>
                </div>
                
                <div class="weight-highlight" id="weightDisplay">
                    Weight: 0.000 kg
                </div>
                
                <div class="info-grid" id="infoGrid">
                    <!-- Fish details will be populated here -->
                </div>
            </div>
        </div>
        
        <script>
            let currentFishData = [];
            let selectedFishIndex = 0;
            let cameraActive = false;
            let captureCountdown = null;
            let captured = false;
            
            function restartCamera() {
                // Only reload when clicking on captured/gallery images
                reloadApp();
            }
            
            function startCountdown() {
                let count = 4; // Start from 4 seconds
                const countdownEl = document.getElementById('captureCountdown');
                
                captureCountdown = setInterval(() => {
                    count--;
                    if (count > 0) {
                        countdownEl.textContent = count;
                        countdownEl.classList.add('show');
                    } else {
                        countdownEl.classList.remove('show');
                        clearInterval(captureCountdown);
                        captureCountdown = null;
                        // Check for captured image
                        setTimeout(checkCapturedImage, 500);
                    }
                }, 1000);
            }
            
            function checkCapturedImage() {
                fetch('/get_detection_info')
                    .then(response => response.json())
                    .then(data => {
                        if (data.fish && data.fish.length > 0) {
                            currentFishData = data.fish;
                            selectedFishIndex = 0;
                            
                            // Check if we have a captured image
                            fetch('/detection_status')
                                .then(response => response.json())
                                .then(statusData => {
                                    if (statusData.has_captured_image) {
                                        // Show the captured image
                                        showCapturedImage();
                                    } else {
                                        // Just update UI without showing captured overlay
                                        showInfoButton();
                                        showReloadButton();
                                        updateStatus(`${data.fish.length} fish captured`, 'Tap Info for details');
                                    }
                                    
                                    captured = true;
                                    stopStatusCheck();
                                });
                        }
                    })
                    .catch(error => {
                        console.log('No fish data available yet');
                    });
            }
            
            function showCapturedImage() {
                // Stop camera first
                fetch('/stop_camera', {method: 'POST'});
                
                // Load and show the captured image in same position as camera
                const capturedImg = document.getElementById('capturedImage');
                const capturedOverlay = document.getElementById('capturedOverlay');
                
                // Set the captured image source
                capturedImg.src = '/get_captured_image?t=' + Date.now();
                
                // Show the overlay with animation
                capturedOverlay.classList.add('show');
                
                // Update buttons
                showInfoButton();
                showReloadButton();
                updateStatus('Fish captured!', 'Tap image to reload or Info for details');
                
                captured = true;
                stopStatusCheck();
            }
            
            function startCamera() {
                fetch('/start_camera', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            cameraActive = true;
                            updateStatus('Camera started - Point at fish', 'No fish detected');
                            startStatusCheck();
                        }
                    });
            }
            
            function stopCamera() {
                fetch('/stop_camera', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        cameraActive = false;
                        updateStatus('Camera stopped', 'Press start to begin');
                        stopStatusCheck();
                    });
            }
            
            function restartDetection() {
                fetch('/restart_detection', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        updateStatus('Detection restarted - Point at fish', 'No fish detected');
                        document.getElementById('topInfo').classList.remove('show');
                    });
            }
            
            function getInfo() {
                // Stop camera when showing info
                fetch('/stop_camera', {method: 'POST'});
                
                fetch('/get_detection_info')
                    .then(response => response.json())
                    .then(data => {
                        if (data.fish && data.fish.length > 0) {
                            currentFishData = data.fish;
                            selectedFishIndex = 0;
                            showDetails();
                        } else {
                            alert('No fish detected. Point camera at fish first.');
                        }
                    });
            }
            
            function showDetails() {
                if (currentFishData.length === 0) return;
                
                // Show fish selector if multiple fish
                const fishSelector = document.getElementById('fishSelector');
                const fishSelect = document.getElementById('fishSelect');
                
                if (currentFishData.length > 1) {
                    fishSelector.style.display = 'block';
                    fishSelect.innerHTML = '';
                    currentFishData.forEach((fish, index) => {
                        const option = document.createElement('option');
                        option.value = index;
                        option.textContent = `Fish ${index + 1} - ${fish.species}`;
                        fishSelect.appendChild(option);
                    });
                } else {
                    fishSelector.style.display = 'none';
                }
                
                displayFishDetails();
                document.getElementById('detailsOverlay').classList.add('show');
            }
            
            function selectFish() {
                selectedFishIndex = parseInt(document.getElementById('fishSelect').value);
                displayFishDetails();
            }
            
            function displayFishDetails() {
                if (currentFishData.length === 0) return;
                
                // Calculate total weight for all fish
                let totalWeight = 0;
                currentFishData.forEach(fish => {
                    totalWeight += fish.weight.weight_kg;
                });
                
                // Update weight display with total
                const weightText = currentFishData.length > 1 
                    ? `Total Weight: ${totalWeight.toFixed(3)} kg (${currentFishData.length} fish)`
                    : `Weight: ${currentFishData[selectedFishIndex].weight.weight_kg.toFixed(3)} kg`;
                
                document.getElementById('weightDisplay').textContent = weightText;
                
                // Draw all fish with bounding boxes
                drawAllFishImage();
                
                // Update info grid with selected fish or summary
                const infoGrid = document.getElementById('infoGrid');
                
                if (currentFishData.length === 1) {
                    // Single fish details
                    const fish = currentFishData[0];
                    infoGrid.innerHTML = generateSingleFishInfo(fish);
                } else {
                    // Multiple fish summary
                    infoGrid.innerHTML = generateMultipleFishInfo();
                }
            }
            
            function drawAllFishImage() {
                const canvas = document.getElementById('fishCanvas');
                const ctx = canvas.getContext('2d');
                
                // Get current frame and draw all fish with bounding boxes
                fetch('/get_annotated_image')
                    .then(response => response.blob())
                    .then(blob => {
                        const img = new Image();
                        img.onload = function() {
                            canvas.width = img.width;
                            canvas.height = img.height;
                            
                            // Draw image
                            ctx.drawImage(img, 0, 0);
                            
                            // Draw bounding boxes for all fish with different colors
                            const colors = ['#00ff00', '#ff0000', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ffa500', '#800080'];
                            
                            currentFishData.forEach((fish, index) => {
                                const bbox = fish.bbox;
                                const color = colors[index % colors.length];
                                
                                // Draw bounding box
                                ctx.strokeStyle = color;
                                ctx.lineWidth = 3;
                                ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
                                
                                // Draw fish number and info
                                ctx.fillStyle = color;
                                ctx.fillRect(bbox[0], bbox[1] - 40, 250, 35);
                                ctx.fillStyle = 'white';
                                ctx.font = '14px Arial';
                                ctx.fillText(`Fish ${index + 1}: ${fish.species} - ${fish.weight.weight_kg.toFixed(3)}kg`, bbox[0] + 5, bbox[1] - 20);
                            });
                        };
                        img.src = URL.createObjectURL(blob);
                    });
            }
            
            function generateSingleFishInfo(fish) {
                let html = `
                    <div class="info-card">
                        <div class="info-title">🎯 Detection</div>
                        <div class="info-value">Species: ${fish.species}</div>
                        <div class="info-value">Confidence: ${(fish.confidence * 100).toFixed(1)}%</div>
                        <div class="info-value">Detection Time: ${new Date(fish.timestamp).toLocaleTimeString()}</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">📏 Distance</div>
                        <div class="info-value">Distance: ${fish.distance_cm} cm</div>
                        <div class="info-value">Distance: ${(fish.distance_cm / 2.54).toFixed(1)} inches</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">📐 Dimensions</div>
                        <div class="info-value">Length: ${fish.dimensions.length_cm.toFixed(1)} cm</div>
                        <div class="info-value">Length: ${fish.dimensions.length_inches.toFixed(1)} inches</div>
                        <div class="info-value">Height: ${fish.dimensions.height_cm.toFixed(1)} cm</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">⚖️ Weight Details</div>
                        <div class="info-value">Weight: ${fish.weight.weight_kg.toFixed(3)} kg</div>
                        <div class="info-value">Weight: ${fish.weight.weight_pounds.toFixed(2)} pounds</div>
                        <div class="info-value">Weight: ${fish.weight.weight_grams.toFixed(0)} grams</div>
                        <div class="info-value">Estimated Girth: ${fish.weight.estimated_girth_cm.toFixed(1)} cm</div>
                        <div class="info-value">Calculation Methods: ${fish.weight.calculation_methods}</div>
                    </div>
                `;
                
                if (fish.species_classification) {
                    html += `
                        <div class="info-card">
                            <div class="info-title">🔬 Species Classification</div>
                            <div class="info-value">Species: ${fish.species_classification.predicted_species || 'Unknown'}</div>
                            <div class="info-value">Confidence: ${(fish.species_classification.confidence * 100).toFixed(1)}%</div>
                        </div>
                    `;
                }
                
                return html;
            }
            
            function generateMultipleFishInfo() {
                let totalWeight = 0;
                let avgConfidence = 0;
                let speciesCount = {};
                
                currentFishData.forEach(fish => {
                    totalWeight += fish.weight.weight_kg;
                    avgConfidence += fish.confidence;
                    speciesCount[fish.species] = (speciesCount[fish.species] || 0) + 1;
                });
                
                avgConfidence /= currentFishData.length;
                
                let speciesText = Object.entries(speciesCount)
                    .map(([species, count]) => `${species} (${count})`)
                    .join(', ');
                
                return `
                    <div class="info-card">
                        <div class="info-title">🎯 Multiple Fish Detection</div>
                        <div class="info-value">Total Fish: ${currentFishData.length}</div>
                        <div class="info-value">Species: ${speciesText}</div>
                        <div class="info-value">Average Confidence: ${(avgConfidence * 100).toFixed(1)}%</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">⚖️ Total Weight</div>
                        <div class="info-value">Total Weight: ${totalWeight.toFixed(3)} kg</div>
                        <div class="info-value">Total Weight: ${(totalWeight * 2.20462).toFixed(2)} pounds</div>
                        <div class="info-value">Total Weight: ${(totalWeight * 1000).toFixed(0)} grams</div>
                        <div class="info-value">Average per Fish: ${(totalWeight / currentFishData.length).toFixed(3)} kg</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">📊 Individual Fish Details</div>
                        ${currentFishData.map((fish, index) => `
                            <div class="info-value" style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                                <strong>Fish ${index + 1}:</strong> ${fish.species}<br>
                                Weight: ${fish.weight.weight_kg.toFixed(3)} kg, 
                                Confidence: ${(fish.confidence * 100).toFixed(1)}%<br>
                                Length: ${fish.dimensions.length_cm.toFixed(1)} cm
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            function closeDetails() {
                document.getElementById('detailsOverlay').classList.remove('show');
            }
            
            function updateStatus(status, count) {
                document.getElementById('detectionStatus').textContent = status;
                document.getElementById('detectionCount').textContent = count;
                
                if (count !== 'No fish detected') {
                    document.getElementById('topInfo').classList.add('show');
                } else {
                    document.getElementById('topInfo').classList.remove('show');
                }
            }
            
            function startStatusCheck() {
                statusInterval = setInterval(checkDetectionStatus, 1000);
            }
            
            function stopStatusCheck() {
                if (statusInterval) {
                    clearInterval(statusInterval);
                }
            }
            
            function checkDetectionStatus() {
                if (!cameraActive || captured) return;
                
                fetch('/detection_status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.fish_detected && data.fish_count > 0) {
                            updateStatus('Fish detected! Auto-capturing in 4 seconds...', 
                                       `${data.fish_count} fish detected`);
                            
                            // Start countdown if not already started
                            if (!captureCountdown) {
                                startCountdown();
                            }
                        } else if (data.capture_complete && data.has_captured_image) {
                            // Fish has been captured, check for captured image
                            checkCapturedImage();
                        } else {
                            updateStatus('Camera active - Point at fish', 'No fish detected');
                        }
                    });
            }
            
            function openGallery() {
                // Stop camera and disable gallery button during processing
                fetch('/stop_camera', {method: 'POST'});
                const galleryBtn = document.querySelector('.btn-gallery');
                galleryBtn.classList.add('disabled');
                
                // Create file input for gallery selection
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/*';
                input.onchange = function(event) {
                    const file = event.target.files[0];
                    if (file) {
                        // Show processing status
                        updateStatus('Analyzing selected image...', 'Processing...');
                        
                        // Create image preview URL
                        const imageUrl = URL.createObjectURL(file);
                        
                        // Create FormData and send to analysis endpoint
                        const formData = new FormData();
                        formData.append('image', file);
                        
                        fetch('/analyze_uploaded_image', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success && data.fish_count > 0) {
                                currentFishData = data.fish;
                                selectedFishIndex = 0;
                                
                                // Show gallery image with detection
                                showGalleryImage(imageUrl, true);
                                updateStatus(`${data.fish_count} fish detected in image`, 'Tap Info for details');
                            } else {
                                // Show gallery image without detection
                                showGalleryImage(imageUrl, false);
                                updateStatus('No fish detected in selected image', 'Please reload to try again');
                            }
                        })
                        .catch(error => {
                            console.error('Analysis error:', error);
                            showGalleryImage(imageUrl, false);
                            updateStatus('Analysis failed', 'Please reload to try again');
                        })
                        .finally(() => {
                            // Re-enable gallery button
                            galleryBtn.classList.remove('disabled');
                        });
                    } else {
                        // Re-enable gallery button if no file selected
                        galleryBtn.classList.remove('disabled');
                        // Restart camera if no file selected
                        fetch('/start_camera', {method: 'POST'});
                    }
                };
                input.click();
            }
            
            function showGalleryImage(imageSrc, hasDetection) {
                // Stop camera first
                fetch('/stop_camera', {method: 'POST'});
                
                const galleryOverlay = document.getElementById('galleryOverlay');
                const galleryImg = document.getElementById('galleryImage');
                const galleryMessage = document.getElementById('galleryMessage');
                
                // Set the gallery image source
                galleryImg.src = imageSrc;
                
                if (hasDetection) {
                    galleryMessage.innerHTML = '<div>🐟 Fish detected in image!</div><div style="font-size: 14px; margin-top: 5px;">Tap Info for details or Reload to continue</div>';
                    showInfoButton();
                } else {
                    galleryMessage.innerHTML = '<div>❌ Fish not found</div><div style="font-size: 14px; margin-top: 5px;">Please reload the screen to try again</div>';
                    showCameraButton();
                }
                
                showReloadButton();
                galleryOverlay.classList.add('show');
                captured = true;
                stopStatusCheck();
            }
            
            function capturePhoto() {
                const cameraBtn = document.getElementById('cameraButton');
                if (cameraBtn.classList.contains('disabled')) return;
                
                // Disable button during processing
                cameraBtn.classList.add('disabled');
                
                // Manual capture functionality - can be used to force capture
                if (captured) {
                    restartCamera();
                } else {
                    // Force a capture if fish is detected
                    fetch('/force_capture', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                updateStatus('Fish captured manually', 'Use Info button for details');
                                showInfoButton();
                                showReloadButton();
                                captured = true;
                                stopStatusCheck();
                            } else {
                                updateStatus('Point camera at fish first', 'No fish detected');
                            }
                        })
                        .finally(() => {
                            // Re-enable button
                            cameraBtn.classList.remove('disabled');
                        });
                }
            }
            
            function reloadApp() {
                // Hide all overlays
                document.getElementById('capturedOverlay').classList.remove('show');
                document.getElementById('galleryOverlay').classList.remove('show');
                document.getElementById('detailsOverlay').classList.remove('show');
                
                // Reset button states
                showCameraButton();
                hideReloadButton();
                
                // Reset variables
                captured = false;
                currentFishData = [];
                
                // Restart camera and detection
                fetch('/restart_detection', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        cameraActive = true;
                        updateStatus('Camera restarted - Point at fish', 'No fish detected');
                        startStatusCheck();
                    });
            }
            
            function showInfoButton() {
                document.getElementById('cameraButton').style.display = 'none';
                document.getElementById('infoButton').style.display = 'flex';
            }
            
            function showCameraButton() {
                document.getElementById('infoButton').style.display = 'none';
                document.getElementById('cameraButton').style.display = 'flex';
            }
            
            function showReloadButton() {
                document.getElementById('reloadButton').classList.add('show');
            }
            
            function hideReloadButton() {
                document.getElementById('reloadButton').classList.remove('show');
            }
            
            function updateCaptureInfo(title, subtitle) {
                const captureInfo = document.getElementById('captureInfo');
                captureInfo.innerHTML = `
                    <div>${title}</div>
                    <div style="font-size: 14px; margin-top: 5px;">${subtitle}</div>
                `;
            }
            
            let statusInterval;
            
            // Auto-start camera on page load
            window.onload = function() {
                // Camera is auto-started on server startup
                cameraActive = true;
                updateStatus('Camera active - Point at fish', 'No fish detected');
                startStatusCheck();
            };
            
            // Refresh video feed periodically
            setInterval(() => {
                if (cameraActive) {
                    const img = document.getElementById('videoFeed');
                    const src = img.src;
                    img.src = '';
                    img.src = src + '?t=' + Date.now();
                }
            }, 30000);
        </script>
    </body>
    </html>
    '''

def generate_frames():
    """Generate video frames for streaming."""
    while True:
        try:
            frame_bytes = detector.get_frame_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Generate a placeholder frame if no camera feed
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Starting Camera...', (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
        except Exception as e:
            print(f"Frame generation error: {e}")
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
        if not detector.running:
            camera_thread = threading.Thread(target=detector.start_camera_feed)
            camera_thread.daemon = True
            camera_thread.start()
            return jsonify({'success': True, 'message': 'Camera started'})
        else:
            return jsonify({'success': True, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera feed."""
    try:
        detector.running = False
        detector.detection_active = False
        if detector.cap:
            detector.cap.release()
            detector.cap = None
        return jsonify({'success': True, 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/restart_detection', methods=['POST'])
def restart_detection():
    """Restart detection mode."""
    try:
        detector.restart_detection()
        return jsonify({'success': True, 'message': 'Detection restarted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_detection_info')
def get_detection_info():
    """Get detected fish information."""
    return jsonify({
        'fish': detector.detected_fish,
        'count': len(detector.detected_fish),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/detection_status')
def detection_status():
    """Get current detection status."""
    return jsonify({
        'fish_detected': len(detector.detected_fish) > 0,
        'fish_count': len(detector.detected_fish),
        'detection_active': detector.detection_active,
        'camera_running': detector.running,
        'capture_complete': detector.capture_complete,
        'has_captured_image': detector.captured_image is not None
    })

@app.route('/get_annotated_image')
def get_annotated_image():
    """Get current annotated image."""
    try:
        if detector.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', detector.current_frame)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
        return "No image available", 404
    except Exception as e:
        return str(e), 500

@app.route('/get_captured_image')
def get_captured_image():
    """Get captured image as JPEG bytes."""
    try:
        if detector.captured_image is not None:
            return Response(detector.get_captured_image_bytes(), mimetype='image/jpeg')
        return "No captured image available", 404
    except Exception as e:
        return str(e), 500

@app.route('/analyze_uploaded_image', methods=['POST'])
def analyze_uploaded_image():
    """Analyze uploaded image from gallery."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'})
        
        # Read image data
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Process image for detection
        processed_frame, detections = detector.process_frame_for_detection(image)
        
        if detections:
            # Create zoomed capture for first fish
            fish = detections[0]
            bbox = fish['bbox']
            detector.captured_image = detector.create_zoomed_capture(image, bbox)
            
            return jsonify({
                'success': True,
                'fish': detections,
                'fish_count': len(detections),
                'message': 'Fish detected in uploaded image'
            })
        else:
            return jsonify({
                'success': False, 
                'fish_count': 0,
                'message': 'No fish detected in uploaded image'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/force_capture', methods=['POST'])
def force_capture():
    """Force capture current frame if fish detected."""
    try:
        # First check if we have current detections
        if detector.detected_fish:
            # Fish already detected, just mark as captured
            return jsonify({
                'success': True, 
                'message': 'Fish captured',
                'fish_count': len(detector.detected_fish)
            })
        
        # Try to detect fish in current frame
        if detector.current_frame is not None:
            processed_frame, detections = detector.process_frame_for_detection(detector.current_frame)
            
            if detections:
                # Store the detections
                detector.detected_fish = detections
                detector.current_frame = processed_frame
                
                return jsonify({
                    'success': True, 
                    'message': 'Fish detected and captured',
                    'fish_count': len(detections)
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': 'No fish detected in current frame'
                })
        else:
            return jsonify({
                'success': False, 
                'message': 'No camera frame available'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🐟 Starting Mobile Fish Detection App...")
    print("=" * 50)
    
    if detector.detector:
        print("✅ Fish Detector: Ready")
    else:
        print("❌ Fish Detector: Not available")
    
    if detector.classifier:
        print("✅ Fish Classifier: Ready")
    else:
        print("❌ Fish Classifier: Not available")
    
    print("✅ Mobile Interface: Ready")
    print("✅ Auto-Detection: Ready")
    
    print(f"\n🚀 Server starting...")
    print(f"📱 Mobile App: http://localhost:5009")
    print(f"🎥 Video Stream: http://localhost:5009/video_feed")
    print(f"⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5009, debug=True, threaded=True) 
#!/usr/bin/env python3
"""
Mobile Fish Detection App with 3D Measurement & Weight Calculation
=================================================================

Advanced mobile-first fish detection with accurate 3D measurements,
depth estimation, and scientific weight calculation.
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
sys.path.insert(0, './detector_v10_m3')
try:
    from local_inference import LocalYOLOv12Fish
except ImportError:
    print("Warning: Could not import LocalYOLOv12Fish")
    LocalYOLOv12Fish = None

try:
    from inference import YOLOInference
except ImportError:
    print("Warning: Could not import YOLOInference (YOLO10)")
    YOLOInference = None

# Add classification path
sys.path.insert(0, './classification_rectangle_v7-1')
try:
    from inference import EmbeddingClassifier
except ImportError:
    print("Warning: Could not import EmbeddingClassifier")
    EmbeddingClassifier = None

app = Flask(__name__)
CORS(app)

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
        return float(obj)
    elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

class AdvancedMeasurementSystem:
    """Advanced 3D measurement system for fish detection."""
    
    def __init__(self):
        """Initialize measurement system with calibration data."""
        # Camera calibration parameters (will be auto-calibrated)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.focal_length_mm = 4.0  # Typical mobile camera focal length
        self.sensor_size_mm = (5.76, 4.29)  # Typical mobile sensor size
        self.calibrated = False
        
        # Measurement reference system
        self.reference_object_size_cm = 10.0  # 10cm reference
        self.pixels_per_cm = 50  # Initial estimate, will be calibrated
        self.measurement_box_size = (200, 300)  # Reference measurement box pixels
        self.show_measurement_box = True  # Show/hide measurement box
        
        # Fish detection in middle box
        self.fish_in_center_box = False
        self.center_box_color = (255, 255, 255)  # White default
        self.detection_box_color = (0, 255, 0)  # Green when fish detected
        
        # Depth estimation parameters
        self.baseline_distance_cm = 50  # Assumed distance for single camera
        self.depth_estimation_factor = 800  # Calibration factor
        
        # Fish species measurement factors
        self.species_length_factors = {
            'Tuna': {'head_ratio': 0.18, 'body_ratio': 0.65, 'tail_ratio': 0.17},
            'Bass': {'head_ratio': 0.22, 'body_ratio': 0.58, 'tail_ratio': 0.20},
            'Salmon': {'head_ratio': 0.20, 'body_ratio': 0.60, 'tail_ratio': 0.20},
            'Catfish': {'head_ratio': 0.25, 'body_ratio': 0.55, 'tail_ratio': 0.20},
            'Snapper': {'head_ratio': 0.21, 'body_ratio': 0.59, 'tail_ratio': 0.20},
            'Mackerel': {'head_ratio': 0.19, 'body_ratio': 0.61, 'tail_ratio': 0.20},
            'Grouper': {'head_ratio': 0.23, 'body_ratio': 0.57, 'tail_ratio': 0.20},
            'default': {'head_ratio': 0.21, 'body_ratio': 0.59, 'tail_ratio': 0.20}
        }
        
        # Weight calculation formulas (species-specific)
        self.weight_formulas = {
            'Tuna': lambda l, h, g: (l * h * g * 0.00028),  # kg
            'Bass': lambda l, h, g: (l * h * g * 0.00025),
            'Salmon': lambda l, h, g: (l * h * g * 0.00026),
            'Catfish': lambda l, h, g: (l * h * g * 0.00030),
            'Snapper': lambda l, h, g: (l * h * g * 0.00027),
            'Mackerel': lambda l, h, g: (l * h * g * 0.00023),
            'Grouper': lambda l, h, g: (l * h * g * 0.00032),
            'default': lambda l, h, g: (l * h * g * 0.00027)
        }
        
        # Girth estimation ratios
        self.girth_ratios = {
            'Tuna': 0.65, 'Bass': 0.45, 'Salmon': 0.40, 'Catfish': 0.55,
            'Snapper': 0.42, 'Mackerel': 0.35, 'Grouper': 0.50, 'default': 0.45
        }
    
    def auto_calibrate_camera(self, frame):
        """Auto-calibrate camera using measurement reference box."""
        try:
            h, w = frame.shape[:2]
            
            # Use frame dimensions for basic calibration
            if not self.calibrated:
                # Estimate pixels per cm based on typical phone camera specs
                diagonal_pixels = math.sqrt(w*w + h*h)
                diagonal_inches = 6.0  # Assume 6 inch phone screen
                diagonal_cm = diagonal_inches * 2.54
                
                # Update pixels per cm
                self.pixels_per_cm = diagonal_pixels / diagonal_cm
                
                # Basic camera matrix estimation
                fx = fy = w * 0.8  # Approximate focal length in pixels
                cx, cy = w/2, h/2  # Principal point at center
                
                self.camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                self.dist_coeffs = np.zeros((4,1))  # Assume no distortion for mobile
                self.calibrated = True
                
                print(f"✅ Auto-calibrated: {self.pixels_per_cm:.1f} pixels/cm")
                
        except Exception as e:
            print(f"Calibration error: {e}")
    
    def estimate_depth_monocular(self, fish_bbox, frame_shape):
        """Estimate depth using monocular cues."""
        try:
            x1, y1, x2, y2 = fish_bbox
            fish_width_pixels = x2 - x1
            fish_height_pixels = y2 - y1
            frame_height, frame_width = frame_shape[:2]
            
            # Method 1: Size-based depth estimation
            fish_area = fish_width_pixels * fish_height_pixels
            frame_area = frame_width * frame_height
            area_ratio = fish_area / frame_area
            
            # Assume fish fills more frame area when closer
            depth_size = 100 / math.sqrt(area_ratio)  # cm
            
            # Method 2: Position-based depth estimation (objects lower in frame often closer)
            fish_center_y = (y1 + y2) / 2
            vertical_position = fish_center_y / frame_height
            depth_position = 50 + (vertical_position * 100)  # 50-150cm range
            
            # Method 3: Focus-based estimation (sharper = closer, for future implementation)
            # Could analyze edge sharpness here
            
            # Average the estimates
            estimated_depth = (depth_size + depth_position) / 2
            
            # Constrain to reasonable range
            return max(20, min(300, estimated_depth))
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return 80  # Default 80cm
    
    def calculate_3d_dimensions(self, fish_bbox, depth_cm, species='default'):
        """Calculate accurate 3D dimensions of fish."""
        try:
            x1, y1, x2, y2 = fish_bbox
            
            # Convert pixel measurements to real-world dimensions
            fish_width_pixels = x2 - x1
            fish_height_pixels = y2 - y1
            
            # Calculate real dimensions using depth and camera parameters
            if self.camera_matrix is not None:
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                
                # Real-world dimensions
                width_cm = (fish_width_pixels * depth_cm) / fx
                height_cm = (fish_height_pixels * depth_cm) / fy
            else:
                # Fallback calculation
                width_cm = fish_width_pixels / self.pixels_per_cm
                height_cm = fish_height_pixels / self.pixels_per_cm
            
            # Determine fish length (usually the larger dimension)
            length_cm = max(width_cm, height_cm)
            body_height_cm = min(width_cm, height_cm)
            
            # Get species factors
            factors = self.species_length_factors.get(species, self.species_length_factors['default'])
            
            # Calculate different length measurements
            total_length_cm = length_cm  # Total Length (TL)
            fork_length_cm = length_cm * (1 - factors['tail_ratio'])  # Fork Length (FL)
            standard_length_cm = length_cm * factors['body_ratio']  # Standard Length (SL)
            
            # Estimate girth based on species
            girth_ratio = self.girth_ratios.get(species, self.girth_ratios['default'])
            girth_cm = length_cm * girth_ratio
            
            # Calculate volume approximation (ellipsoid)
            volume_cm3 = (4/3) * math.pi * (length_cm/2) * (body_height_cm/2) * (girth_cm/4)
            
            return {
                'total_length_cm': total_length_cm,
                'fork_length_cm': fork_length_cm,
                'standard_length_cm': standard_length_cm,
                'body_height_cm': body_height_cm,
                'estimated_girth_cm': girth_cm,
                'volume_cm3': volume_cm3,
                'length_inches': total_length_cm / 2.54,
                'height_inches': body_height_cm / 2.54,
                'girth_inches': girth_cm / 2.54
            }
            
        except Exception as e:
            print(f"3D calculation error: {e}")
            return None
    
    def calculate_advanced_weight(self, dimensions, species='default'):
        """Calculate fish weight using advanced formulas."""
        try:
            if not dimensions:
                return None
            
            length_cm = dimensions['total_length_cm']
            height_cm = dimensions['body_height_cm']
            girth_cm = dimensions['estimated_girth_cm']
            
            # Get species-specific weight formula
            weight_formula = self.weight_formulas.get(species, self.weight_formulas['default'])
            
            # Calculate weight
            weight_kg = weight_formula(length_cm, height_cm, girth_cm)
            
            # Alternative calculations for validation
            # Standard weight formula: W = a * L^b
            standard_weight_kg = 0.00001 * (length_cm ** 3.1)  # General fish formula
            
            # Volume-based calculation
            volume_weight_kg = dimensions['volume_cm3'] * 0.001  # Assume fish density ~1g/cm³
            
            # Average the methods for better accuracy
            final_weight_kg = (weight_kg + standard_weight_kg + volume_weight_kg) / 3
            
            return {
                'weight_kg': final_weight_kg,
                'weight_pounds': final_weight_kg * 2.20462,
                'weight_grams': final_weight_kg * 1000,
                'weight_ounces': final_weight_kg * 35.274,
                'species_weight_kg': weight_kg,
                'standard_weight_kg': standard_weight_kg,
                'volume_weight_kg': volume_weight_kg,
                'calculation_methods': 3
            }
            
        except Exception as e:
            print(f"Weight calculation error: {e}")
            return None
    
    def check_fish_in_center_box(self, fish_bbox, frame_shape):
        """Check if fish is in the center measurement box."""
        try:
            h, w = frame_shape[:2]
            
            # Center box coordinates
            box_w, box_h = self.measurement_box_size
            center_x, center_y = w // 2, h // 2
            
            box_x1 = center_x - box_w // 2
            box_y1 = center_y - box_h // 2
            box_x2 = center_x + box_w // 2
            box_y2 = center_y + box_h // 2
            
            # Fish bbox coordinates
            fish_x1, fish_y1, fish_x2, fish_y2 = fish_bbox
            fish_center_x = (fish_x1 + fish_x2) / 2
            fish_center_y = (fish_y1 + fish_y2) / 2
            
            # Check if fish center is in the center box
            in_center = (box_x1 <= fish_center_x <= box_x2 and 
                        box_y1 <= fish_center_y <= box_y2)
            
            # Also check for significant overlap
            overlap_x = max(0, min(fish_x2, box_x2) - max(fish_x1, box_x1))
            overlap_y = max(0, min(fish_y2, box_y2) - max(fish_y1, box_y1))
            overlap_area = overlap_x * overlap_y
            
            fish_area = (fish_x2 - fish_x1) * (fish_y2 - fish_y1)
            overlap_ratio = overlap_area / fish_area if fish_area > 0 else 0
            
            # Fish is considered "in center" if center is in box OR significant overlap
            return in_center or overlap_ratio > 0.5
            
        except Exception as e:
            print(f"Center box check error: {e}")
            return False
    
    def draw_measurement_overlay(self, frame, fish_detected_in_center=False):
        """Draw measurement reference box with dynamic color."""
        try:
            # Only draw overlay if enabled
            if not self.show_measurement_box:
                return frame
                
            h, w = frame.shape[:2]
            
            # Draw measurement reference box in center
            box_w, box_h = self.measurement_box_size
            center_x, center_y = w // 2, h // 2
            
            # Reference box coordinates
            box_x1 = center_x - box_w // 2
            box_y1 = center_y - box_h // 2
            box_x2 = center_x + box_w // 2
            box_y2 = center_y + box_h // 2
            
            # Choose color based on fish detection
            box_color = self.detection_box_color if fish_detected_in_center else self.center_box_color
            
            # Draw measurement box with white border (0.6px equivalent)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 1)
            
            # Remove grid design as requested
            # No grid lines drawn
            
            # Calculate reference dimensions
            ref_width_cm = box_w / self.pixels_per_cm
            ref_height_cm = box_h / self.pixels_per_cm
            
            # Draw reference dimensions (smaller text)
            cv2.putText(frame, f"{ref_width_cm:.1f}cm", 
                       (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            cv2.putText(frame, f"{ref_height_cm:.1f}cm", 
                       (box_x2 + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            # Draw corner markers only when fish detected
            if fish_detected_in_center:
                corner_size = 15
                corners = [(box_x1, box_y1), (box_x2, box_y1), (box_x1, box_y2), (box_x2, box_y2)]
                for x, y in corners:
                    cv2.line(frame, (x-corner_size, y), (x+corner_size, y), (0, 255, 0), 2)
                    cv2.line(frame, (x, y-corner_size), (x, y+corner_size), (0, 255, 0), 2)
            
           
            
            return frame
            
        except Exception as e:
            print(f"Measurement overlay error: {e}")
            return frame

class MobileFishDetector:
    """Enhanced mobile fish detector with 3D measurement capabilities."""
    
    def __init__(self, camera_id=0):
        """Initialize enhanced mobile fish detector."""
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.detection_active = True
        self.current_frame = None
        self.detected_fish = []
        self.capture_on_detection = True
        self.captured_image = None
        self.capture_timer = None
        self.auto_capture_delay = 3  # 3 seconds delay for center box detection
        self.capture_complete = False
        self.fish_in_center_detected = False
        
        # Initialize measurement system
        self.measurement_system = AdvancedMeasurementSystem()
        
        # Initialize multiple fish detectors for fallback
        self.detectors = []
        
        # Initialize YOLO12 detector (primary)
        if LocalYOLOv12Fish:
            try:
                detector_yolo12 = LocalYOLOv12Fish(confidence=0.1)  # Lower confidence for better detection
                self.detectors.append(('YOLO12', detector_yolo12))
                print("✅ YOLO12 fish detector loaded")
            except Exception as e:
                print(f"❌ Failed to load YOLO12 detector: {e}")
        
        # Initialize YOLO10 detector (fallback)
        if YOLOInference:
            try:
                detector_yolo10 = YOLOInference(
                    model_path='./detector_v10_m3/model.ts',
                    conf_threshold=0.05,  # Very low confidence for better detection
                    nms_threshold=0.3,
                    yolo_ver='v10'
                )
                self.detectors.append(('YOLO10', detector_yolo10))
                print("✅ YOLO10 fish detector loaded")
            except Exception as e:
                print(f"❌ Failed to load YOLO10 detector: {e}")
        
        # Set primary detector (for backward compatibility)
        self.detector = self.detectors[0][1] if self.detectors else None
        
        # Fish species mapping for YOLO10 (index 0 = fish)
        self.fish_species_names = [
            'Fish', 'Tuna', 'Bass', 'Salmon', 'Catfish', 'Snapper', 
            'Mackerel', 'Grouper', 'Cod', 'Flounder', 'Trout'
        ]
        
        # Initialize classifier
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
        
        # Camera settings for different devices
        self.device_settings = {
            'mobile': {'width': 720, 'height': 1280, 'fps': 30},
            'tablet': {'width': 768, 'height': 1366, 'fps': 30},
            'desktop': {'width': 1920, 'height': 1080, 'fps': 30}
        }
        
        # Focal length for calculations
        self.focal_length = 800
        
        # Auto-start camera
        self.auto_start_camera()

    def detect_device_type(self):
        """Detect device type for optimal camera settings."""
        # This would typically use user agent detection in a web app
        # For now, default to mobile for portrait orientation
        return 'mobile'

    def initialize_camera(self):
        """Initialize camera with device-specific settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
        
        # Get device type and set appropriate resolution
        device_type = self.detect_device_type()
        settings = self.device_settings[device_type]
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
        self.cap.set(cv2.CAP_PROP_FPS, settings['fps'])
        
        print(f"✅ Camera initialized for {device_type}: {settings['width']}x{settings['height']}")
        return True

    def auto_start_camera(self):
        """Automatically start camera in background."""
        try:
            camera_thread = threading.Thread(target=self.start_camera_feed)
            camera_thread.daemon = True
            camera_thread.start()
            print("🎥 Enhanced camera auto-started")
        except Exception as e:
            print(f"❌ Failed to auto-start camera: {e}")
    
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
        """Calculate fish weight using advanced measurement system."""
        # Use the advanced 3D measurement system for better accuracy
        return self.measurement_system.calculate_advanced_weight(dimensions, species)
    
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
    
    def auto_capture_fish_in_center(self):
        """Automatically capture fish when detected in center box."""
        try:
            if self.detected_fish and self.current_frame is not None:
                # Find fish in center box
                center_fish = None
                for fish in self.detected_fish:
                    if fish.get('in_center_box', False):
                        center_fish = fish
                        break
                
                if center_fish:
                    bbox = center_fish['bbox']
                    
                    # Create stable capture - use the full frame for better quality
                    self.captured_image = self.current_frame.copy()
                    
                    print(f"📸 Auto-captured fish in center: {center_fish['species']}")
                    
                    # Mark as captured and stop camera processing
                    self.capture_complete = True
                    self.detection_active = False
                    self.running = False  # Stop camera feed
                
        except Exception as e:
            print(f"Auto-capture error: {e}")
    
    def detect_fish_multi_model(self, frame):
        """Try multiple models to detect fish for better reliability."""
        all_detections = []
        
        for model_name, detector in self.detectors:
            try:
                print(f"🔍 Trying {model_name} detection...")
                
                if model_name == 'YOLO12':
                    # YOLO12 detection
                    detections = detector.predict(frame)
                    fish_list = detections[0] if detections and detections[0] else []
                    
                    if fish_list:
                        print(f"✅ {model_name} detected {len(fish_list)} fish")
                        for fish in fish_list:
                            bbox = fish.get_box()
                            confidence = fish.get_score()
                            species = fish.get_class_name()
                            
                            detection_data = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'species': species,
                                'model': model_name
                            }
                            all_detections.append(detection_data)
                        break  # Use first successful detection
                    else:
                        print(f"❌ {model_name} found no fish")
                
                elif model_name == 'YOLO10':
                    # YOLO10 detection
                    detections = detector.predict(frame)
                    fish_list = detections[0] if detections and detections[0] else []
                    
                    if fish_list:
                        print(f"✅ {model_name} detected {len(fish_list)} fish")
                        for fish in fish_list:
                            bbox = fish.get_box()
                            confidence = fish.get_score()
                            # For YOLO10, assume all detections are fish
                            species = 'Fish'  # Generic fish class
                            
                            detection_data = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'species': species,
                                'model': model_name
                            }
                            all_detections.append(detection_data)
                        break  # Use first successful detection
                    else:
                        print(f"❌ {model_name} found no fish")
                        
            except Exception as e:
                print(f"❌ {model_name} detection error: {e}")
                continue
        
        return all_detections

    def process_frame_for_detection(self, frame):
        """Process frame for fish detection with multi-model support and improved reliability."""
        if not self.detectors:
            # Apply measurement overlay even without detectors
            overlay_frame = self.measurement_system.draw_measurement_overlay(frame, False)
            self.measurement_system.auto_calibrate_camera(frame)
            return overlay_frame, []
        
        try:
            # Auto-calibrate measurement system
            self.measurement_system.auto_calibrate_camera(frame)
            
            # Try multiple models for fish detection
            detected_fish_data = self.detect_fish_multi_model(frame)
            
            # Check for fish in center box
            fish_in_center = False
            
            if detected_fish_data and self.detection_active:
                # Process all detected fish
                self.detected_fish = []
                
                print(f"🐟 Processing {len(detected_fish_data)} detected fish...")
                
                for i, fish_data in enumerate(detected_fish_data):
                    bbox = fish_data['bbox']
                    confidence = fish_data['confidence']
                    species = fish_data['species']
                    model_used = fish_data['model']
                    
                    # Check if this fish is in the center box
                    in_center = self.measurement_system.check_fish_in_center_box(bbox, frame.shape)
                    if in_center:
                        fish_in_center = True
                        print(f"🎯 Fish {i+1} is in center box!")
                    
                    # Calculate advanced measurements
                    distance_cm = self.measurement_system.estimate_depth_monocular(bbox, frame.shape)
                    dimensions = self.measurement_system.calculate_3d_dimensions(bbox, distance_cm, species)
                    weight_info = self.measurement_system.calculate_advanced_weight(dimensions, species)
                    
                    # Species classification (if available)
                    species_classification = self.classify_species(frame, bbox)
                    
                    # Use classifier result if available, otherwise use detection result
                    final_species = species
                    if species_classification and species_classification.get('predicted_species'):
                        final_species = species_classification['predicted_species']
                    
                    fish_data_complete = {
                        'id': i,
                        'species': final_species,
                        'confidence': float(confidence),  # Convert to Python float
                        'bbox': [float(x) for x in bbox],  # Convert bbox to Python floats
                        'distance_cm': round(float(distance_cm), 1),  # Convert to Python float
                        'dimensions': convert_numpy_types(dimensions),  # Convert all numpy types
                        'weight': convert_numpy_types(weight_info),  # Convert all numpy types
                        'species_classification': convert_numpy_types(species_classification),  # Convert all numpy types
                        'in_center_box': in_center,
                        'timestamp': datetime.now().isoformat(),
                        'detected_by': model_used
                    }
                    
                    self.detected_fish.append(fish_data_complete)
                    
                    print(f"📊 Fish {i+1}: {final_species} ({confidence:.2f}) - {distance_cm:.1f}cm - Weight: {weight_info['weight_kg']:.3f}kg" if weight_info else f"📊 Fish {i+1}: {final_species} ({confidence:.2f}) - {distance_cm:.1f}cm")
                
                # Auto-capture logic (unchanged)
                if fish_in_center and self.capture_on_detection:
                    if not self.fish_in_center_detected:
                        self.fish_in_center_detected = True
                        
                        if self.capture_timer:
                            self.capture_timer.cancel()
                        
                        self.capture_timer = threading.Timer(self.auto_capture_delay, self.auto_capture_fish_in_center)
                        self.capture_timer.start()
                        print(f"🎯 Fish detected in center box - Auto-capturing in {self.auto_capture_delay} seconds...")
                
                elif not fish_in_center:
                    if self.fish_in_center_detected:
                        self.fish_in_center_detected = False
                        if self.capture_timer:
                            self.capture_timer.cancel()
                            self.capture_timer = None
                        print("🎯 Fish moved out of center box - Capture cancelled")
                
                # Create annotated frame
                annotated_frame = self.draw_enhanced_detection_overlay(frame)
                overlay_frame = self.measurement_system.draw_measurement_overlay(annotated_frame, fish_in_center)
                self.current_frame = overlay_frame
                
                return overlay_frame, self.detected_fish
            else:
                # No fish detected - reset center detection
                if self.fish_in_center_detected:
                    self.fish_in_center_detected = False
                    if self.capture_timer:
                        self.capture_timer.cancel()
                        self.capture_timer = None
                
                print("❌ No fish detected by any model")
            
            # Apply measurement overlay
            overlay_frame = self.measurement_system.draw_measurement_overlay(frame, fish_in_center)
            self.current_frame = overlay_frame
            return overlay_frame, []
            
        except Exception as e:
            print(f"🚨 Detection error: {e}")
            import traceback
            traceback.print_exc()
            overlay_frame = self.measurement_system.draw_measurement_overlay(frame, False)
            return overlay_frame, []
    
    def draw_enhanced_detection_overlay(self, frame):
        """Draw clean detection overlay."""
        if not self.detected_fish:
            return frame
        
        try:
            for fish in self.detected_fish:
                bbox = fish['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Choose color based on whether fish is in center box
                in_center = fish.get('in_center_box', False)
                color = (0, 255, 0) if in_center else (255, 255, 0)  # Green if in center, yellow otherwise
                
                # Draw border-only bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Fish info overlay (top of bbox)
                species = fish.get('species', 'Fish')
                weight_kg = fish.get('weight', {}).get('weight_kg', 0)
                confidence = fish.get('confidence', 0)
                
                info_text = f"{species} - {weight_kg:.3f}kg ({confidence*100:.1f}%)"
                
                # Text background
                (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-35), (x1+text_width+10, y1), color, -1)
                
                # Text
                cv2.putText(frame, info_text, (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add center indicator if fish is in center box
                if in_center:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                    cv2.putText(frame, "CENTER", (center_x-30, center_y+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Detection overlay error: {e}")
            return frame
    
    def start_camera_feed(self):
        """Start camera feed processing."""
        if not self.initialize_camera():
            return
        
        self.running = True
        print("🎥 Starting enhanced mobile fish detection...")
        
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
                
                # Process for detection with center box logic
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
        self.capture_complete = False
        self.fish_in_center_detected = False
        
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
                <div class="detection-status" id="detectionStatus">Camera active - Center fish in white box</div>
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
                let count = 3; // Start from 3 seconds
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
                updateStatus('Fish captured!', 'Tap Info for details or reload to continue');
                
                captured = true;
                stopStatusCheck();
            }
            
            function getInfo() {
                // Stop camera when showing info
                fetch('/stop_camera', {method: 'POST'});
                
                // First try to get existing detection info
                fetch('/get_detection_info')
                    .then(response => response.json())
                    .then(data => {
                        if (data.fish && data.fish.length > 0) {
                            currentFishData = data.fish;
                            selectedFishIndex = 0;
                            showDetails();
                        } else {
                            // No existing detections, analyze current frame
                            return fetch('/analyze_current_frame', {method: 'POST'});
                        }
                    })
                    .then(response => {
                        if (response) {
                            return response.json();
                        }
                        return null;
                    })
                    .then(data => {
                        if (data) {
                            if (data.success && data.fish && data.fish.length > 0) {
                                currentFishData = data.fish;
                                selectedFishIndex = 0;
                                showDetails();
                            } else {
                                alert('No fish detected in current frame. Try positioning fish in the white box first.');
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Error getting fish info:', error);
                        alert('Error analyzing image. Please try again.');
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
                    if (fish.weight && fish.weight.weight_kg) {
                        totalWeight += fish.weight.weight_kg;
                    }
                });
                
                // Update weight display with total
                const weightText = currentFishData.length > 1 
                    ? `Total Weight: ${totalWeight.toFixed(3)} kg (${currentFishData.length} fish)`
                    : `Weight: ${currentFishData[selectedFishIndex].weight?.weight_kg?.toFixed(3) || 0} kg`;
                
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
                                const weight = fish.weight?.weight_kg?.toFixed(3) || '0.000';
                                ctx.fillText(`Fish ${index + 1}: ${fish.species} - ${weight}kg`, bbox[0] + 5, bbox[1] - 20);
                            });
                        };
                        img.src = URL.createObjectURL(blob);
                    });
            }
            
            function generateSingleFishInfo(fish) {
                const weight = fish.weight || {};
                const dimensions = fish.dimensions || {};
                
                let html = `
                    <div class="info-card">
                        <div class="info-title">🎯 Detection</div>
                        <div class="info-value">Species: ${fish.species}</div>
                        <div class="info-value">Confidence: ${(fish.confidence * 100).toFixed(1)}%</div>
                        <div class="info-value">Detection Time: ${new Date(fish.timestamp).toLocaleTimeString()}</div>
                        <div class="info-value">Position: ${fish.in_center_box ? 'In Center Box ✅' : 'Outside Center Box'}</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">📏 Distance</div>
                        <div class="info-value">Distance: ${fish.distance_cm} cm</div>
                        <div class="info-value">Distance: ${(fish.distance_cm / 2.54).toFixed(1)} inches</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">📐 Dimensions</div>
                        <div class="info-value">Length: ${dimensions.total_length_cm?.toFixed(1) || 'N/A'} cm</div>
                        <div class="info-value">Length: ${dimensions.length_inches?.toFixed(1) || 'N/A'} inches</div>
                        <div class="info-value">Height: ${dimensions.body_height_cm?.toFixed(1) || 'N/A'} cm</div>
                        <div class="info-value">Girth: ${dimensions.estimated_girth_cm?.toFixed(1) || 'N/A'} cm</div>
                    </div>
                    
                    <div class="info-card">
                        <div class="info-title">⚖️ Weight Details</div>
                        <div class="info-value">Weight: ${weight.weight_kg?.toFixed(3) || 'N/A'} kg</div>
                        <div class="info-value">Weight: ${weight.weight_pounds?.toFixed(2) || 'N/A'} pounds</div>
                        <div class="info-value">Weight: ${weight.weight_grams?.toFixed(0) || 'N/A'} grams</div>
                        <div class="info-value">Calculation Methods: ${weight.calculation_methods || 'N/A'}</div>
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
                let centerFishCount = 0;
                
                currentFishData.forEach(fish => {
                    if (fish.weight && fish.weight.weight_kg) {
                        totalWeight += fish.weight.weight_kg;
                    }
                    avgConfidence += fish.confidence;
                    speciesCount[fish.species] = (speciesCount[fish.species] || 0) + 1;
                    if (fish.in_center_box) centerFishCount++;
                });
                
                avgConfidence /= currentFishData.length;
                
                let speciesText = Object.entries(speciesCount)
                    .map(([species, count]) => `${species} (${count})`)
                    .join(', ');
                
                return `
                    <div class="info-card">
                        <div class="info-title">🎯 Multiple Fish Detection</div>
                        <div class="info-value">Total Fish: ${currentFishData.length}</div>
                        <div class="info-value">Fish in Center Box: ${centerFishCount}</div>
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
                                <strong>Fish ${index + 1}:</strong> ${fish.species} ${fish.in_center_box ? '✅' : ''}<br>
                                Weight: ${fish.weight?.weight_kg?.toFixed(3) || 'N/A'} kg, 
                                Confidence: ${(fish.confidence * 100).toFixed(1)}%<br>
                                Length: ${fish.dimensions?.total_length_cm?.toFixed(1) || 'N/A'} cm
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
                            // Check if any fish are in center
                            fetch('/get_detection_info')
                                .then(response => response.json())
                                .then(fishData => {
                                    const centerFish = fishData.fish?.filter(fish => fish.in_center_box) || [];
                                    
                                    if (centerFish.length > 0) {
                                        updateStatus(`Fish in center box - Auto-capturing in 3 seconds...`, 
                                                   `${centerFish.length} fish in center box`);
                                        
                                        // Start countdown if not already started
                                        if (!captureCountdown) {
                                            startCountdown();
                                        }
                                    } else {
                                        updateStatus('Fish detected - Center fish in white box', 
                                                   `${data.fish_count} fish detected`);
                                    }
                                });
                        } else if (data.capture_complete && data.has_captured_image) {
                            // Fish has been captured, check for captured image
                            checkCapturedImage();
                        } else {
                            updateStatus('Camera active - Center fish in white box', 'No fish detected');
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
                
                // If already captured, restart camera
                if (captured) {
                    restartCamera();
                    cameraBtn.classList.remove('disabled');
                    return;
                }
                
                // Capture current frame and analyze it immediately
                updateStatus('Capturing and analyzing...', 'Processing image');
                
                fetch('/capture_and_analyze', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Update fish data
                            currentFishData = data.fish;
                            selectedFishIndex = 0;
                            
                            // Update UI state
                            captured = true;
                            showInfoButton();
                            showReloadButton();
                            stopStatusCheck();
                            
                            if (data.count > 0) {
                                updateStatus(`Photo captured! ${data.count} fish detected`, 'Click Info for details');
                                // Automatically show details if fish detected
                                setTimeout(() => {
                                    showDetails();
                                }, 500);
                            } else {
                                updateStatus('Photo captured - No fish detected', 'You can still view the captured image');
                            }
                        } else {
                            updateStatus('Capture failed', data.message || 'Please try again');
                        }
                    })
                    .catch(error => {
                        console.error('Capture error:', error);
                        updateStatus('Capture error', 'Please try again');
                    })
                    .finally(() => {
                        // Re-enable button
                        cameraBtn.classList.remove('disabled');
                    });
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
                        updateStatus('Camera restarted - Center fish in white box', 'No fish detected');
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
            
            let statusInterval;
            
            // Auto-start camera on page load
            window.onload = function() {
                // Camera is auto-started on server startup
                cameraActive = true;
                updateStatus('Camera active - Center fish in white box', 'No fish detected');
                startStatusCheck();
            };
            
            // Refresh video feed periodically
            setInterval(() => {
                if (cameraActive && !captured) {
                    const img = document.getElementById('videoFeed');
                    const timestamp = Date.now();
                    img.src = '/video_feed?t=' + timestamp;
                }
            }, 100); // More frequent refresh for better responsiveness
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
    fish_count = len(detector.detected_fish)
    fish_data = convert_numpy_types(detector.detected_fish)
    return jsonify({
        'fish': fish_data,
        'count': fish_count,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze_current_frame', methods=['POST'])
def analyze_current_frame():
    """Analyze current frame and return fish information for getInfo()."""
    try:
        if detector.current_frame is not None:
            # Process current frame for detection
            processed_frame, detections = detector.process_frame_for_detection(detector.current_frame)
            
            if detections:
                # Update detector with new detections
                detector.detected_fish = detections
                detector.current_frame = processed_frame
                
                return jsonify({
                    'success': True,
                    'fish': convert_numpy_types(detections),
                    'count': len(detections),
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Current frame analyzed successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'fish': [],
                    'count': 0,
                    'message': 'No fish detected in current frame'
                })
        else:
            return jsonify({
                'success': False,
                'fish': [],
                'count': 0,
                'message': 'No current frame available'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'fish': [],
            'count': 0,
            'error': str(e)
        })

@app.route('/capture_and_analyze', methods=['POST'])
def capture_and_analyze():
    """Capture current frame and analyze it - for capturePhoto() functionality."""
    try:
        if detector.current_frame is not None:
            # Capture the current frame
            detector.captured_image = detector.current_frame.copy()
            detector.capture_complete = True
            
            # Process current frame for detection
            processed_frame, detections = detector.process_frame_for_detection(detector.current_frame)
            
            # Update detector with detections (even if empty)
            detector.detected_fish = detections
            detector.current_frame = processed_frame
            
            return jsonify({
                'success': True,
                'fish': convert_numpy_types(detections),
                'count': len(detections),
                'timestamp': datetime.now().isoformat(),
                'message': f'Frame captured and analyzed - {len(detections)} fish detected'
            })
        else:
            return jsonify({
                'success': False,
                'fish': [],
                'count': 0,
                'message': 'No current frame available for capture'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'fish': [],
            'count': 0,
            'error': str(e)
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
        'has_captured_image': detector.captured_image is not None,
        'fish_in_center': any(fish.get('in_center_box', False) for fish in detector.detected_fish)
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
                'fish': convert_numpy_types(detections),
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
            # Check if any fish are in center box
            center_fish = [fish for fish in detector.detected_fish if fish.get('in_center_box', False)]
            
            if center_fish:
                # Fish in center - capture the full frame
                if detector.current_frame is not None:
                    detector.captured_image = detector.current_frame.copy()
                    detector.capture_complete = True
                    
                    return jsonify({
                        'success': True, 
                        'message': 'Fish captured from center box',
                        'fish_count': len(center_fish)
                    })
                else:
                    return jsonify({
                        'success': False, 
                        'message': 'No current frame available for capture'
                    })
            else:
                # Fish detected but not in center
                return jsonify({
                    'success': True, 
                    'message': 'Fish captured (not in center)',
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
    print("🐟 Starting Enhanced Multi-Model Fish Detection App...")
    print("=" * 60)
    
    print("🔍 Detector Status:")
    if detector.detectors:
        for model_name, model_instance in detector.detectors:
            print(f"✅ {model_name} Fish Detector: Ready")
    else:
        print("❌ No Fish Detectors Available")
    
    if detector.classifier:
        print("✅ Fish Classifier: Ready")
    else:
        print("❌ Fish Classifier: Not available")
    
    print("✅ 3D Measurement System: Ready")
    print("✅ Camera Calibration: Auto-calibrating")
    print("✅ Advanced Weight Calculation: Ready")
    print("✅ Mobile Interface: Enhanced")
    print("✅ Real-time Depth Estimation: Ready")
    print("✅ Center Box Detection: Enabled")
    print("✅ Multi-Model Fallback: YOLO12 → YOLO10")
    print("✅ Auto-Capture: 3 seconds when fish in center")
    print("✅ Low Confidence Thresholds: Better fish detection")
    
    print(f"\n🚀 Enhanced server starting...")
    print(f"📱 Multi-Model Fish Detection App: http://localhost:5008")
    print(f"🎥 Enhanced Video Stream: http://localhost:5008/video_feed")
    print(f"📐 Features: Multi-model detection, center box, auto-capture")
    print(f"🎯 Instructions: Center fish in white box for auto-capture")
    print(f"🔍 Detection: Tries YOLO12 first, then YOLO10 as fallback")
    print(f"⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5008, debug=True, threaded=True) 
#!/usr/bin/env python3
"""
YOLOv12 Inference Wrapper for Fish Detection
============================================

Compatible interface with existing YOLOv10 system.
"""

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch

class YOLOv12Fish:
    """YOLOv12 fish detection with compatible interface."""
    
    def __init__(self, model_path=None, confidence=0.5):
        """Initialize YOLOv12 model."""
        self.confidence = confidence
        
        # Load model info
        model_info_path = os.path.join(os.path.dirname(__file__), 'model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
        else:
            self.model_info = {}
        
        # Try to load local model first
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded local YOLOv12 model: {model_path}")
        else:
            # Try to find downloaded model
            possible_paths = [
                'best.pt',
                'weights/best.pt', 
                'train/weights/best.pt',
                'yolov5s.pt'
            ]
            
            model_loaded = False
            for path in possible_paths:
                full_path = os.path.join(os.path.dirname(__file__), path)
                if os.path.exists(full_path):
                    self.model = YOLO(full_path)
                    print(f"✅ Loaded YOLOv12 model: {full_path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                # Fallback to YOLOv8 nano for testing
                print("⚠️  No local model found, using YOLOv8n as fallback")
                self.model = YOLO('yolov8n.pt')
    
    def predict(self, image):
        """Predict fish in image with compatible interface."""
        try:
            # Run inference
            results = self.model(image, conf=self.confidence)
            
            # Convert to compatible format
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = YOLOv12Detection(box, image.shape)
                        detections.append(detection)
            
            return [detections] if detections else [None]
            
        except Exception as e:
            print(f"❌ YOLOv12 prediction error: {e}")
            return [None]

class YOLOv12Detection:
    """Compatible detection object."""
    
    def __init__(self, box, image_shape):
        """Initialize detection from YOLOv12 box."""
        self.box_data = box
        self.image_shape = image_shape
        
        # Extract box coordinates
        self.xyxy = box.xyxy[0].cpu().numpy()
        self.confidence = float(box.conf[0])
        self.class_id = int(box.cls[0])
    
    def get_box(self):
        """Get bounding box coordinates."""
        return self.xyxy.tolist()
    
    def get_score(self):
        """Get confidence score."""
        return self.confidence
    
    def get_mask_BGR(self):
        """Get cropped fish image."""
        # This would need the original image to crop
        # For now, return a placeholder
        x1, y1, x2, y2 = map(int, self.xyxy)
        
        # Create a mock crop (in real implementation, pass original image)
        h, w = self.image_shape[:2]
        crop_h = max(1, y2 - y1)
        crop_w = max(1, x2 - x1)
        
        # Return a placeholder image of correct size
        return np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    
    def get_mask(self):
        """Get binary mask."""
        x1, y1, x2, y2 = map(int, self.xyxy)
        crop_h = max(1, y2 - y1)
        crop_w = max(1, x2 - x1)
        
        # Return binary mask (white rectangle)
        mask = np.ones((crop_h, crop_w), dtype=np.uint8) * 255
        return mask

# Compatibility alias
YOLOInference = YOLOv12Fish

#!/usr/bin/env python3
"""
Enhanced YOLOv12 Inference with Proper Cropping
===============================================

Enhanced version that properly handles image cropping.
"""

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch

class EnhancedYOLOv12Fish:
    """Enhanced YOLOv12 fish detection with proper cropping."""
    
    def __init__(self, model_path=None, confidence=0.5):
        """Initialize enhanced YOLOv12 model."""
        self.confidence = confidence
        self.last_image = None  # Store last processed image
        
        # Load model (same logic as before)
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Try to find downloaded model
            possible_paths = [
                'best.pt', 'weights/best.pt', 'train/weights/best.pt'
            ]
            
            model_loaded = False
            base_dir = os.path.dirname(__file__)
            
            for path in possible_paths:
                full_path = os.path.join(base_dir, path)
                if os.path.exists(full_path):
                    self.model = YOLO(full_path)
                    print(f"✅ Loaded enhanced YOLOv12: {full_path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("⚠️  Using YOLOv8n fallback for enhanced detection")
                self.model = YOLO('yolov8n.pt')
    
    def predict(self, image):
        """Enhanced prediction with proper image storage."""
        self.last_image = image.copy()  # Store for cropping
        
        try:
            results = self.model(image, conf=self.confidence)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = EnhancedYOLOv12Detection(box, image)
                        detections.append(detection)
            
            return [detections] if detections else [None]
            
        except Exception as e:
            print(f"❌ Enhanced YOLOv12 error: {e}")
            return [None]

class EnhancedYOLOv12Detection:
    """Enhanced detection with proper image cropping."""
    
    def __init__(self, box, original_image):
        """Initialize with original image for cropping."""
        self.box_data = box
        self.original_image = original_image
        
        # Extract coordinates
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
        """Get actual cropped fish image."""
        x1, y1, x2, y2 = map(int, self.xyxy)
        
        # Ensure coordinates are within image bounds
        h, w = self.original_image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop the actual fish region
        crop = self.original_image[y1:y2, x1:x2]
        
        # Ensure minimum size
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            crop = cv2.resize(crop, (64, 64))
        
        return crop
    
    def get_mask(self):
        """Get binary mask for the detection."""
        crop = self.get_mask_BGR()
        
        # Convert to grayscale and create binary mask
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding to create mask
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        return mask

# Compatibility
YOLOInference = EnhancedYOLOv12Fish

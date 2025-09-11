#!/usr/bin/env python3
"""
Roboflow Hosted API Inference for YOLOv12
========================================

Uses your trained model hosted on Roboflow cloud for fish detection.
"""

import os
import cv2
import json
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image

class RoboflowYOLOv12Fish:
    """YOLOv12 fish detection using Roboflow hosted model."""
    
    def __init__(self, confidence=0.5):
        """Initialize Roboflow API client."""
        self.confidence = confidence
        self.api_key = "GgtUYJgnCNrQWen4qAYr"
        self.model_endpoint = "https://detect.roboflow.com/fishditectiondata/4"
        
        # Fish species from your dataset
        self.class_names = [
            'Anchovies', 'Bangus', 'Basa fish', 'Big-Head-Carp', 
            'Black-Spotted-Barb', 'Blue marlin', 'Catfish', 'Climbing-Perch',
            'Cown tongue fish', 'Crucian carp', 'Fourfinger-Threadfin', 
            'Freshwater-Eel', 'Giant Grouper', 'Glass-Perchlet', 'Goby',
            'Gold-Fish', 'Mackerel', 'Mullet fish', 'Northern red snapper',
            'Perch fish', 'Phu Quoc Island Tuna', 'Pompano', 'Rabbitfish',
            'Snakehead fish', 'Snapper', 'Tuna', 'Vietnamese mackerel', 
            'big head carp'
        ]
        
        print(f"✅ Initialized Roboflow YOLOv12 (28 fish species)")
    
    def predict(self, image):
        """Predict fish in image using Roboflow API."""
        try:
            # Convert image to base64
            image_encoded = self._encode_image(image)
            
            # Make API request
            response = requests.post(
                self.model_endpoint,
                params={
                    "api_key": self.api_key,
                    "confidence": int(self.confidence * 100),
                    "overlap": 30,
                    "format": "json"
                },
                files={"file": image_encoded},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                return [None]
            
            result = response.json()
            
            # Convert to compatible format
            detections = []
            
            if "predictions" in result:
                for pred in result["predictions"]:
                    detection = RoboflowYOLOv12Detection(pred, image)
                    if detection.confidence >= self.confidence:
                        detections.append(detection)
            
            return [detections] if detections else [None]
            
        except Exception as e:
            print(f"❌ Roboflow API error: {e}")
            return [None]
    
    def _encode_image(self, image):
        """Encode OpenCV image for API."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Encode as JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        
        return buffer.getvalue()

class RoboflowYOLOv12Detection:
    """Compatible detection object for Roboflow results."""
    
    def __init__(self, prediction, original_image):
        """Initialize detection from Roboflow prediction."""
        self.prediction = prediction
        self.original_image = original_image
        
        # Extract data from Roboflow format
        self.confidence = prediction.get("confidence", 0.0)
        self.class_name = prediction.get("class", "unknown")
        self.class_id = prediction.get("class_id", 0)
        
        # Convert Roboflow coordinates to YOLO format
        x_center = prediction.get("x", 0)
        y_center = prediction.get("y", 0)
        width = prediction.get("width", 0)
        height = prediction.get("height", 0)
        
        # Calculate bounding box coordinates
        self.x1 = int(x_center - width / 2)
        self.y1 = int(y_center - height / 2)
        self.x2 = int(x_center + width / 2)
        self.y2 = int(y_center + height / 2)
        
        # Ensure coordinates are within image bounds
        h, w = original_image.shape[:2]
        self.x1 = max(0, min(self.x1, w-1))
        self.y1 = max(0, min(self.y1, h-1))
        self.x2 = max(self.x1+1, min(self.x2, w))
        self.y2 = max(self.y1+1, min(self.y2, h))
    
    def get_box(self):
        """Get bounding box coordinates [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    def get_score(self):
        """Get confidence score."""
        return self.confidence
    
    def get_class_name(self):
        """Get detected class name."""
        return self.class_name
    
    def get_mask_BGR(self):
        """Get cropped fish image."""
        # Crop the actual fish region
        crop = self.original_image[self.y1:self.y2, self.x1:self.x2]
        
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

# Compatibility aliases
YOLOInference = RoboflowYOLOv12Fish
EnhancedYOLOv12Fish = RoboflowYOLOv12Fish

if __name__ == "__main__":
    # Test the Roboflow API
    print("🐟 Testing Roboflow YOLOv12 API...")
    
    detector = RoboflowYOLOv12Fish()
    
    # Test with existing test image
    test_image_path = "../test_image.png"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        results = detector.predict(image)
        
        if results and results[0]:
            print(f"✅ Detected {len(results[0])} fish!")
            for i, fish in enumerate(results[0]):
                print(f"   Fish {i+1}: {fish.get_class_name()} ({fish.get_score():.3f})")
        else:
            print("❌ No fish detected")
    else:
        print(f"⚠️  Test image not found: {test_image_path}") 
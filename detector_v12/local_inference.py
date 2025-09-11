#!/usr/bin/env python3
"""
Local YOLOv12 Inference using Trained Weights
============================================

Uses your locally trained YOLOv12 model for fish detection.
"""

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch

class LocalYOLOv12Fish:
    """YOLOv12 fish detection using locally trained model."""
    
    def __init__(self, model_path=None, confidence=0.5):
        """Initialize local YOLOv12 model."""
        self.confidence = confidence
        
        # Use the trained model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Loaded local YOLOv12 model: {model_path}")
            
            # Fish species from your dataset (28 classes)
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
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def predict(self, image):
        """Predict fish in image using local model."""
        try:
            # Run inference
            results = self.model(image, conf=self.confidence)
            
            # Convert to compatible format
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = LocalYOLOv12Detection(box, image, self.class_names)
                        detections.append(detection)
            
            return [detections] if detections else [None]
            
        except Exception as e:
            print(f"❌ Local YOLOv12 prediction error: {e}")
            return [None]

class LocalYOLOv12Detection:
    """Compatible detection object for local model results."""
    
    def __init__(self, box, original_image, class_names):
        """Initialize detection from local model box."""
        self.box_data = box
        self.original_image = original_image
        self.class_names = class_names
        
        # Extract data
        self.xyxy = box.xyxy[0].cpu().numpy()
        self.confidence = float(box.conf[0])
        self.class_id = int(box.cls[0])
        
        # Get class name
        if 0 <= self.class_id < len(self.class_names):
            self.class_name = self.class_names[self.class_id]
        else:
            self.class_name = f"class_{self.class_id}"
        
        # Calculate coordinates
        self.x1, self.y1, self.x2, self.y2 = map(int, self.xyxy)
        
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
YOLOInference = LocalYOLOv12Fish
EnhancedYOLOv12Fish = LocalYOLOv12Fish

if __name__ == "__main__":
    # Test the local trained model
    print("🐟 Testing Local YOLOv12 Model...")
    
    try:
        detector = LocalYOLOv12Fish()
        
        # Test with existing test image
        test_image_path = "../test_image.png"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            results = detector.predict(image)
            
            if results and results[0]:
                print(f"✅ Local model detected {len(results[0])} fish!")
                for i, fish in enumerate(results[0]):
                    print(f"   Fish {i+1}: {fish.get_class_name()} ({fish.get_score():.3f})")
            else:
                print("❌ No fish detected")
        else:
            print(f"⚠️  Test image not found: {test_image_path}")
            
    except Exception as e:
        print(f"❌ Local model test failed: {e}") 
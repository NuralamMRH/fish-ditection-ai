#!/usr/bin/env python3
"""
Download YOLOv12 Trained Model Weights from Roboflow
===================================================

This script downloads the actual trained model weights from your Roboflow project.
"""

import os
import requests
import json
from roboflow import Roboflow

def download_trained_model():
    """Download the trained model weights from Roboflow."""
    print("📥 Downloading trained YOLOv12 model weights...")
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key="GgtUYJgnCNrQWen4qAYr")
        project = rf.workspace("rancoded").project("fishditectiondata")
        version = project.version(4)
        
        print(f"📋 Project: {project.name}")
        print(f"📦 Version: {version.version}")
        
        # Get the model
        model = version.model
        print(f"🎯 Model ID: {model.id}")
        
        # Try to download model in different formats
        formats_to_try = [
            "yolov8",      # YOLOv8 format (most compatible)
            "yolov5",      # YOLOv5 format
            "yolov11",     # Latest format
            "ultralytics"  # Generic ultralytics format
        ]
        
        model_downloaded = False
        
        for format_name in formats_to_try:
            try:
                print(f"🔄 Trying to download in {format_name} format...")
                dataset = version.download(format_name)
                
                # Look for model weights
                possible_weights = [
                    "best.pt",
                    "weights/best.pt",
                    "train/weights/best.pt",
                    "runs/train/weights/best.pt",
                    "yolov8n.pt",
                    "model.pt"
                ]
                
                for weight_path in possible_weights:
                    full_path = os.path.join(dataset.location, weight_path)
                    if os.path.exists(full_path):
                        # Copy to detector_v12
                        import shutil
                        dest_path = os.path.join("detector_v12", "best.pt")
                        shutil.copy2(full_path, dest_path)
                        
                        print(f"✅ Model weights found and copied: {weight_path}")
                        print(f"✅ Saved to: {dest_path}")
                        
                        # Update model info
                        model_info = {
                            "model_id": model.id,
                            "version": version.version,
                            "format": format_name,
                            "weights_path": dest_path,
                            "local_path": dataset.location,
                            "endpoint": getattr(model, 'api_url', None),
                            "classes": 28,
                            "class_names": [
                                'Anchovies', 'Bangus', 'Basa fish', 'Big-Head-Carp', 
                                'Black-Spotted-Barb', 'Blue marlin', 'Catfish', 'Climbing-Perch',
                                'Cown tongue fish', 'Crucian carp', 'Fourfinger-Threadfin', 
                                'Freshwater-Eel', 'Giant Grouper', 'Glass-Perchlet', 'Goby',
                                'Gold-Fish', 'Mackerel', 'Mullet fish', 'Northern red snapper',
                                'Perch fish', 'Phu Quoc Island Tuna', 'Pompano', 'Rabbitfish',
                                'Snakehead fish', 'Snapper', 'Tuna', 'Vietnamese mackerel', 
                                'big head carp'
                            ]
                        }
                        
                        with open('detector_v12/model_info.json', 'w') as f:
                            json.dump(model_info, f, indent=2)
                        
                        model_downloaded = True
                        break
                
                if model_downloaded:
                    break
                    
            except Exception as e:
                print(f"❌ Failed to download {format_name}: {e}")
                continue
        
        if not model_downloaded:
            print("⚠️  Could not find trained weights. Trying Roboflow hosted model...")
            
            # Create API-based model info for hosted inference
            model_info = {
                "model_id": model.id,
                "version": version.version,
                "format": "hosted_api",
                "weights_path": None,
                "endpoint": f"https://detect.roboflow.com/fishditectiondata/4",
                "api_key": "GgtUYJgnCNrQWen4qAYr",
                "classes": 28,
                "class_names": [
                    'Anchovies', 'Bangus', 'Basa fish', 'Big-Head-Carp', 
                    'Black-Spotted-Barb', 'Blue marlin', 'Catfish', 'Climbing-Perch',
                    'Cown tongue fish', 'Crucian carp', 'Fourfinger-Threadfin', 
                    'Freshwater-Eel', 'Giant Grouper', 'Glass-Perchlet', 'Goby',
                    'Gold-Fish', 'Mackerel', 'Mullet fish', 'Northern red snapper',
                    'Perch fish', 'Phu Quoc Island Tuna', 'Pompano', 'Rabbitfish',
                    'Snakehead fish', 'Snapper', 'Tuna', 'Vietnamese mackerel', 
                    'big head carp'
                ]
            }
            
            with open('detector_v12/model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print("✅ Configured for Roboflow hosted API inference")
        
        return model_downloaded
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def train_new_model():
    """Train a new YOLOv8 model on the downloaded dataset."""
    print("🏋️ Training new YOLOv8 model on your dataset...")
    
    try:
        from ultralytics import YOLO
        
        # Load a pretrained YOLOv8 model
        model = YOLO('yolov8n.pt')  # Start with nano for faster training
        
        # Train the model
        print("🔄 Starting training (this may take a while)...")
        results = model.train(
            data='detector_v12/data.yaml',
            epochs=50,              # Adjust based on your needs
            imgsz=640,             # Image size
            batch=16,              # Batch size (adjust based on your GPU)
            device='cpu',          # Change to 'cuda' if you have GPU
            patience=10,           # Early stopping patience
            save=True,
            exist_ok=True,
            project='detector_v12',
            name='train'
        )
        
        # Copy the best weights
        best_model_path = 'detector_v12/train/weights/best.pt'
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, 'detector_v12/best.pt')
            print(f"✅ Training complete! Model saved to: detector_v12/best.pt")
            return True
        else:
            print("❌ Training completed but no weights found")
            return False
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("🐟 YOLOv12 Model Weight Download & Training")
    print("=" * 50)
    
    # Try to download trained weights first
    if download_trained_model():
        print("✅ Downloaded trained model weights!")
    else:
        print("\n⚠️  No trained weights found. Would you like to train a new model?")
        response = input("Train new model? (y/n): ").lower().strip()
        
        if response == 'y':
            if train_new_model():
                print("✅ New model trained successfully!")
            else:
                print("❌ Training failed")
        else:
            print("ℹ️  Using Roboflow hosted API for inference")
    
    print("\n🎯 Ready to test YOLOv12 integration!") 
#!/usr/bin/env python3
"""
Complete Fish Identification Demo
=================================

This script demonstrates how to use the full fish identification pipeline:
1. YOLOv10 Fish Detection - finds fish in images
2. Fish Classification - identifies fish species 
3. Fish Segmentation - creates pixel-level masks

Requirements:
- All model files in their respective directories
- Python dependencies installed (see requirements.txt)
- Test images to process

Author: Fish Identification Project
Website: www.fishial.ai
"""

import os
import sys
from pathlib import Path

def setup_models():
    """Initialize all three models: detection, classification, and segmentation."""
    print("🐟 Setting up Fish Identification Models...")
    
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return None, None, None
    
    # Add model directories to path
    sys.path.append('./detector_v10_m3')
    sys.path.append('./classification_rectangle_v7-1')
    
    try:
        # 1. Setup YOLOv10 Fish Detector
        print("📍 Loading YOLOv10 Fish Detector...")
        from inference import YOLOInference  # From detector_v10_m3
        detector = YOLOInference(
            model_path='./detector_v10_m3/model.ts',
            imsz=(640, 640),
            conf_threshold=0.25,
            nms_threshold=0.45,
            yolo_ver='v10'
        )
        
        # Remove detector path and add classifier path
        sys.path.remove('./detector_v10_m3')
        
        # 2. Setup Fish Classifier (Latest v7-1)
        print("🔍 Loading Fish Classifier...")
        from inference import EmbeddingClassifier  # From classification_rectangle_v7-1
        classifier = EmbeddingClassifier(
            model_path='./classification_rectangle_v7-1/model.ts',
            data_set_path='./classification_rectangle_v7-1/database.pt',
            device='cpu'
        )
        
        # 3. Setup Fish Segmentation (TorchScript)
        print("🎭 Loading Fish Segmentation Model...")
        segmentation_model = torch.jit.load('./segmentation_21_08_2023.ts')
        segmentation_model.eval()
        
        print("✅ All models loaded successfully!")
        return detector, classifier, segmentation_model
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure all model files exist in their directories")
        return None, None, None

def process_image(image_path, detector, classifier, segmentation_model, output_dir="output"):
    """Process a single image through the complete pipeline."""
    
    import cv2
    import numpy as np
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"\n🖼️  Processing: {image_path}")
    original_image = image.copy()
    
    # Step 1: Fish Detection with YOLOv10
    print("1️⃣  Running Fish Detection...")
    detections = detector.predict(image)
    
    if not detections or not detections[0]:
        print("   ❌ No fish detected in image")
        return
    
    fish_results = detections[0]
    print(f"   ✅ Detected {len(fish_results)} fish")
    
    # Step 2: Process each detected fish
    for i, fish in enumerate(fish_results):
        print(f"\n🐠 Processing Fish #{i+1}")
        
        # Get bounding box and crop fish
        box = fish.get_box()  # [x1, y1, x2, y2]
        confidence = fish.get_score()
        
        print(f"   📦 Bounding Box: {box}, Confidence: {confidence:.3f}")
        
        # Crop fish from image
        fish_crop = fish.get_mask_BGR()  # This gets the cropped fish
        
        # Step 3: Fish Classification
        print("   🔍 Classifying fish species...")
        classification_results = classifier.inference_numpy(fish_crop)
        
        if classification_results:
            top_species = classification_results[0]
            species_name = top_species['name']
            accuracy = top_species['accuracy']
            print(f"   🎯 Species: {species_name} (Accuracy: {accuracy:.3f})")
        else:
            species_name = "Unknown"
            accuracy = 0.0
            print("   ❓ Species: Unknown")
        
        # Step 4: Draw results on image
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Draw label
        label = f"Fish #{i+1}: {species_name} ({accuracy:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(original_image, (box[0], box[1] - label_size[1] - 10), 
                     (box[0] + label_size[0], box[1]), (0, 255, 0), -1)
        cv2.putText(original_image, label, (box[0], box[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save individual fish crop
        crop_filename = f"{output_dir}/fish_{i+1}_{species_name.replace(' ', '_')}.jpg"
        cv2.imwrite(crop_filename, fish_crop)
        print(f"   💾 Saved fish crop: {crop_filename}")
    
    # Save final result
    result_filename = f"{output_dir}/result_{Path(image_path).stem}.jpg"
    cv2.imwrite(result_filename, original_image)
    print(f"\n💾 Saved result: {result_filename}")

def demo_with_test_image():
    """Run demo with the provided test image."""
    print("🚀 Starting Fish Identification Demo")
    print("=" * 50)
    
    # Setup models
    detector, classifier, segmentation_model = setup_models()
    
    if detector is None:
        return
    
    # Process test image
    test_image = "test_image.png"
    if os.path.exists(test_image):
        process_image(test_image, detector, classifier, segmentation_model)
    else:
        print(f"❌ Test image not found: {test_image}")
        print("   Please add an image file or update the path")

def demo_with_custom_image(image_path):
    """Run demo with a custom image."""
    print("🚀 Starting Fish Identification Demo")
    print("=" * 50)
    
    # Setup models
    detector, classifier, segmentation_model = setup_models()
    
    if detector is None:
        return
    
    # Process custom image
    if os.path.exists(image_path):
        process_image(image_path, detector, classifier, segmentation_model)
    else:
        print(f"❌ Image not found: {image_path}")

def batch_process_directory(input_dir, output_dir="batch_output"):
    """Process all images in a directory."""
    print("🚀 Starting Batch Fish Identification")
    print("=" * 50)
    
    # Setup models
    detector, classifier, segmentation_model = setup_models()
    
    if detector is None:
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in directory: {input_dir}")
        return
    
    print(f"📂 Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in image_files:
        process_image(str(image_file), detector, classifier, segmentation_model, output_dir)

def main():
    """Main function to run the demo."""
    print("🐟 Fish Identification System")
    print("=" * 40)
    print("Options:")
    print("1. Demo with test image")
    print("2. Process custom image")
    print("3. Batch process directory")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            demo_with_test_image()
            break
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            demo_with_custom_image(image_path)
            break
        elif choice == "3":
            input_dir = input("Enter input directory path: ").strip()
            batch_process_directory(input_dir)
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all model files are present and virtual environment is activated") 
#!/usr/bin/env python3
"""
Simple Fish Identification Demo
===============================

This is a beginner-friendly script to test the fish identification models.
Uses the latest working models from the project.

Author: Fish Identification Project
"""

import os
import sys
import cv2

def test_yolo_detection():
    """Test YOLOv10 fish detection."""
    print("🔍 Testing YOLOv10 Fish Detection...")
    print("-" * 40)
    
    try:
        # Add YOLO model directory to path
        sys.path.insert(0, './detector_v10_m3')
        from inference import YOLOInference
        
        # Load detector
        detector = YOLOInference(
            model_path='./detector_v10_m3/model.ts',
            conf_threshold=0.25,
            nms_threshold=0.45,
            yolo_ver='v10'
        )
        
        # Load test image
        if not os.path.exists('test_image.png'):
            print("❌ test_image.png not found!")
            return False
            
        image = cv2.imread('test_image.png')
        print(f"📷 Loaded image: {image.shape}")
        
        # Detect fish
        detections = detector.predict(image)
        
        if detections and detections[0]:
            print(f"✅ Found {len(detections[0])} fish!")
            
            for i, fish in enumerate(detections[0]):
                box = fish.get_box()
                score = fish.get_score()
                area = fish.get_area()
                print(f"  🐠 Fish {i+1}:")
                print(f"     Box: {box}")
                print(f"     Confidence: {score:.3f}")
                print(f"     Area: {area:.0f} pixels")
                
                # Save cropped fish
                fish_crop = fish.get_mask_BGR()
                crop_filename = f"detected_fish_{i+1}.jpg"
                cv2.imwrite(crop_filename, fish_crop)
                print(f"     💾 Saved: {crop_filename}")
            
            # Draw detections on original image
            result_image = image.copy()
            for i, fish in enumerate(detections[0]):
                box = fish.get_box()
                cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(result_image, f"Fish {i+1}", (box[0], box[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imwrite("detection_result.jpg", result_image)
            print("📸 Saved detection result: detection_result.jpg")
            
        else:
            print("❌ No fish detected")
            return False
            
        print("✅ YOLO Detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ YOLO Detection failed: {e}")
        return False
    finally:
        # Clean up sys.path
        if './detector_v10_m3' in sys.path:
            sys.path.remove('./detector_v10_m3')

def test_fish_classification():
    """Test fish classification with latest model."""
    print("\n🔬 Testing Fish Classification (v7-1)...")
    print("-" * 40)
    
    try:
        # Add classification model directory to path  
        sys.path.insert(0, './classification_rectangle_v7-1')
        from inference import EmbeddingClassifier
        
        # Load classifier
        classifier = EmbeddingClassifier(
            model_path='./classification_rectangle_v7-1/model.ts',
            data_set_path='./classification_rectangle_v7-1/database.pt',
            device='cpu'
        )
        
        # Load test image
        image = cv2.imread('test_image.png')
        
        # Classify
        results = classifier.inference_numpy(image)
        
        if results:
            print(f"🎯 Classification Results (Top 5):")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. {result['name']}")
                print(f"     Species ID: {result['species_id']}")
                print(f"     Accuracy: {result['accuracy']:.3f}")
                print(f"     Times found: {result['times']}")
                print()
                
            # Save result to file
            with open("classification_result.txt", "w") as f:
                f.write("Fish Classification Results\n")
                f.write("=" * 30 + "\n\n")
                for i, result in enumerate(results[:5]):
                    f.write(f"{i+1}. {result['name']}\n")
                    f.write(f"   Species ID: {result['species_id']}\n") 
                    f.write(f"   Accuracy: {result['accuracy']:.3f}\n")
                    f.write(f"   Times found: {result['times']}\n\n")
            
            print("📄 Saved classification results: classification_result.txt")
            
        else:
            print("❌ Classification failed")
            return False
            
        print("✅ Classification test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return False
    finally:
        # Clean up sys.path
        if './classification_rectangle_v7-1' in sys.path:
            sys.path.remove('./classification_rectangle_v7-1')

def main():
    """Run all tests."""
    print("🐟 Fish Identification System - Simple Demo")
    print("=" * 50)
    print("Testing all models with test_image.png")
    print()
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment may not be activated")
        print("   Run: source venv/bin/activate")
        print()
    
    results = {}
    
    # Test YOLO detection
    results['detection'] = test_yolo_detection()
    
    # Test classification
    results['classification'] = test_fish_classification()
    
    # Summary
    print("\n📊 Test Summary:")
    print("-" * 20)
    print(f"🔍 YOLO Detection: {'✅ PASSED' if results['detection'] else '❌ FAILED'}")
    print(f"🔬 Classification: {'✅ PASSED' if results['classification'] else '❌ FAILED'}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! Your fish identification system is working!")
        print("\n📁 Generated files:")
        print("   - detection_result.jpg (image with bounding boxes)")
        print("   - detected_fish_*.jpg (individual fish crops)")
        print("   - classification_result.txt (species identification)")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
        print("   Make sure all model files exist and virtual environment is activated.")

if __name__ == "__main__":
    main() 
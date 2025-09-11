#!/usr/bin/env python3
"""
Test Fish Classification Only
=============================

Standalone test for the fish classification model.
"""

import os
import sys
import cv2

def test_classification_v7():
    """Test the latest classification model v7-1."""
    print("🔬 Testing Fish Classification v7-1...")
    
    # Change to classification directory
    original_dir = os.getcwd()
    
    try:
        os.chdir('./classification_rectangle_v7-1')
        
        # Import the classifier
        from inference import EmbeddingClassifier
        
        # Load classifier
        classifier = EmbeddingClassifier(
            model_path='./model.ts',
            data_set_path='./database.pt',
            device='cpu'
        )
        
        # Load test image
        image = cv2.imread('../test_image.png')
        if image is None:
            print("❌ Could not load test image")
            return False
            
        print(f"📷 Image shape: {image.shape}")
        
        # Classify
        results = classifier.inference_numpy(image)
        
        if results:
            print(f"🎯 Classification Results (Top 3):")
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. Species: {result['name']}")
                print(f"     Species ID: {result['species_id']}")
                print(f"     Accuracy: {result['accuracy']:.3f}")
                if 'times' in result:
                    print(f"     Times found: {result['times']}")
                print()
            
            return True
        else:
            print("❌ No classification results")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    test_classification_v7() 
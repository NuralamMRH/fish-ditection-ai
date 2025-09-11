#!/usr/bin/env python3
"""
Fish Identification - Quick Start
=================================

This script tests all your fish identification capabilities
and shows you what's working and what's available.
"""

import os
import sys
import subprocess
import time

def print_header(text):
    print(f"\n{'='*60}")
    print(f"🐟 {text}")
    print(f"{'='*60}")

def print_step(step, text):
    print(f"\n{step}. {text}")

def check_virtual_env():
    """Check if virtual environment is activated."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
        return True
    else:
        print("❌ Virtual environment not activated")
        print("   Run: source venv/bin/activate")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import cv2
        import torch
        import flask
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def check_models():
    """Check if model files exist."""
    models = [
        "detector_v10_m3/model.ts",
        "classification_rectangle_v7-1/model.ts",
        "classification_rectangle_v7-1/database.pt",
        "test_image.png"
    ]
    
    all_exist = True
    for model in models:
        if os.path.exists(model):
            print(f"✅ {model}")
        else:
            print(f"❌ {model} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_yolo_detection():
    """Test YOLO fish detection."""
    try:
        print("🔍 Testing YOLO detection...")
        result = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
sys.path.insert(0, './detector_v10_m3')
from inference import YOLOInference
detector = YOLOInference('./detector_v10_m3/model.ts')
image = cv2.imread('test_image.png')
detections = detector.predict(image)
if detections and detections[0]:
    print(f'SUCCESS: Found {len(detections[0])} fish with confidence {detections[0][0].get_score():.3f}')
else:
    print('FAIL: No fish detected')
"""
        ], capture_output=True, text=True, timeout=30)
        
        if "SUCCESS" in result.stdout:
            print("✅ YOLO Detection working!")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print("❌ YOLO Detection failed")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ YOLO Detection test failed: {e}")
        return False

def test_classification():
    """Test fish classification."""
    try:
        print("🔬 Testing fish classification...")
        result = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
os.chdir('./classification_rectangle_v7-1')
from inference import EmbeddingClassifier
classifier = EmbeddingClassifier('./model.ts', './database.pt')
image = cv2.imread('../test_image.png')
results = classifier.inference_numpy(image)
if results:
    print(f'SUCCESS: Species {results[0]["name"]} with accuracy {results[0]["accuracy"]:.3f}')
else:
    print('FAIL: No classification results')
"""
        ], capture_output=True, text=True, timeout=30)
        
        if "SUCCESS" in result.stdout:
            print("✅ Classification working!")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print("❌ Classification failed")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def show_available_apps():
    """Show available applications."""
    print("\n🚀 AVAILABLE APPLICATIONS:")
    
    apps = [
        {
            "name": "Simple Demo",
            "file": "simple_demo.py", 
            "description": "Test both YOLO and classification",
            "command": "python simple_demo.py"
        },
        {
            "name": "Web Application",
            "file": "run_web_app.py",
            "description": "Upload images via web browser",
            "command": "python run_web_app.py"
        },
        {
            "name": "API Server", 
            "file": "api_demo.py",
            "description": "REST API for programmatic access",
            "command": "python api_demo.py"
        },
        {
            "name": "Legacy Runner",
            "file": "runner.py",
            "description": "Original classification demo",
            "command": "python runner.py"
        }
    ]
    
    for i, app in enumerate(apps, 1):
        status = "✅" if os.path.exists(app["file"]) else "❌"
        print(f"\n{i}. {status} {app['name']}")
        print(f"   📄 File: {app['file']}")
        print(f"   📝 Description: {app['description']}")
        print(f"   🏃 Run: {app['command']}")

def show_devapi_info():
    """Show information about devapi."""
    print("\n🌐 FISHIAL CLOUD API (devapi):")
    
    if os.path.exists("devapi"):
        print("✅ devapi directory found")
        print("📋 This connects to Fishial.ai's cloud service")
        print("🔑 Requires API key from https://portal.fishial.ai")
        print("⚡ Higher accuracy than local models")
        print("💰 Pay-per-request pricing")
        
        if os.path.exists("devapi/recognize_fish.py"):
            print("\n🧪 Test command:")
            print("   cd devapi")
            print("   python recognize_fish.py --key-id YOUR_KEY --key-secret YOUR_SECRET fishpic.jpg")
    else:
        print("❌ devapi directory not found")

def main():
    print_header("Fish Identification Quick Start")
    
    print("This script checks your setup and shows you what's available.")
    
    # Step 1: Check environment
    print_step(1, "Checking Environment")
    env_ok = check_virtual_env()
    deps_ok = check_dependencies()
    
    # Step 2: Check models
    print_step(2, "Checking Model Files")
    models_ok = check_models()
    
    # Step 3: Test functionality  
    print_step(3, "Testing Fish Detection & Classification")
    if env_ok and deps_ok and models_ok:
        yolo_ok = test_yolo_detection()
        class_ok = test_classification()
    else:
        print("⏭️  Skipping tests due to missing requirements")
        yolo_ok = False
        class_ok = False
    
    # Step 4: Show available applications
    print_step(4, "Available Applications")
    show_available_apps()
    
    # Step 5: Show devapi info
    print_step(5, "Cloud API Option")
    show_devapi_info()
    
    # Summary
    print_header("SUMMARY")
    
    status_items = [
        ("Environment", env_ok),
        ("Dependencies", deps_ok), 
        ("Model Files", models_ok),
        ("YOLO Detection", yolo_ok),
        ("Classification", class_ok)
    ]
    
    for item, status in status_items:
        icon = "✅" if status else "❌"
        print(f"{icon} {item}")
    
    if all(status for _, status in status_items):
        print("\n🎉 EVERYTHING IS WORKING!")
        print("\n🚀 NEXT STEPS:")
        print("   1. Run: python run_web_app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Upload fish images and get results!")
        print("\n💡 TIP: Check DEPLOYMENT_GUIDE.md for production deployment")
    else:
        print("\n⚠️  SOME ISSUES FOUND")
        print("\n🔧 TO FIX:")
        if not env_ok:
            print("   • Activate virtual environment: source venv/bin/activate")
        if not deps_ok:
            print("   • Install dependencies: pip install -r requirements.txt")
        if not models_ok:
            print("   • Check that model files exist in their directories")
        print("\n📖 See GETTING_STARTED.md for detailed setup instructions")

if __name__ == "__main__":
    main() 
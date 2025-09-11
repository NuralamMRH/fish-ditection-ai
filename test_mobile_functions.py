#!/usr/bin/env python3
"""
Test script for Mobile Fish Detector App functions
"""

import requests
import json
import time

BASE_URL = "http://localhost:5007"

def test_endpoints():
    """Test all the endpoints that support getInfo() and capturePhoto()"""
    
    print("🧪 Testing Mobile Fish Detector API Endpoints")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/detection_status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            status = response.json()
            print(f"   Camera running: {status.get('camera_running')}")
            print(f"   Detection active: {status.get('detection_active')}")
        else:
            print("❌ Server not responding properly")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure to run: python mobile_fish_detector_app.py")
        return
    
    print()
    
    # Test 2: Test analyze_current_frame endpoint (used by getInfo)
    print("📸 Testing analyze_current_frame endpoint (getInfo functionality)...")
    try:
        response = requests.post(f"{BASE_URL}/analyze_current_frame", timeout=10)
        result = response.json()
        if result.get('success'):
            print("✅ analyze_current_frame works")
            print(f"   Fish detected: {result.get('fish_count', 0)}")
            if result.get('fish'):
                for i, fish in enumerate(result['fish']):
                    print(f"   Fish {i+1}: {fish.get('species')} - {fish.get('weight', {}).get('weight_kg', 0):.3f}kg")
        else:
            print("⚠️  analyze_current_frame: No fish detected")
            print(f"   Message: {result.get('message')}")
    except Exception as e:
        print(f"❌ analyze_current_frame failed: {e}")
    
    print()
    
    # Test 3: Test capture_and_analyze endpoint (used by capturePhoto)
    print("📷 Testing capture_and_analyze endpoint (capturePhoto functionality)...")
    try:
        response = requests.post(f"{BASE_URL}/capture_and_analyze", timeout=10)
        result = response.json()
        if result.get('success'):
            print("✅ capture_and_analyze works")
            print(f"   Fish captured: {result.get('fish_count', 0)}")
            if result.get('fish'):
                for i, fish in enumerate(result['fish']):
                    print(f"   Fish {i+1}: {fish.get('species')} - {fish.get('weight', {}).get('weight_kg', 0):.3f}kg")
        else:
            print("⚠️  capture_and_analyze: No fish detected")
            print(f"   Message: {result.get('message')}")
    except Exception as e:
        print(f"❌ capture_and_analyze failed: {e}")
    
    print()
    
    # Test 4: Test get_detection_info endpoint
    print("📊 Testing get_detection_info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/get_detection_info", timeout=5)
        result = response.json()
        print(f"✅ get_detection_info works")
        print(f"   Stored fish count: {result.get('count', 0)}")
        if result.get('fish'):
            for i, fish in enumerate(result['fish']):
                print(f"   Fish {i+1}: {fish.get('species')} - {fish.get('weight', {}).get('weight_kg', 0):.3f}kg")
    except Exception as e:
        print(f"❌ get_detection_info failed: {e}")
    
    print()
    print("🎯 Test Summary:")
    print("- getInfo() should now work by first checking existing detections")
    print("- If no existing detections, it calls analyze_current_frame")
    print("- capturePhoto() now always analyzes the current frame")
    print("- Both functions provide better error handling and user feedback")
    print()
    print("📱 To test in browser, go to: http://localhost:5007")

if __name__ == "__main__":
    test_endpoints() 
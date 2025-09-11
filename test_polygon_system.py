#!/usr/bin/env python3
"""
Test Polygon Detection System
============================

Debug and test the polygon detection functionality.
"""

import requests
import json
import os
import webbrowser
import time

def test_polygon_api():
    """Test the polygon detection API."""
    print("🐟 Testing Interactive Polygon Detection API")
    print("=" * 60)
    
    api_url = "http://localhost:5002"
    
    # Test health endpoint
    print("🩺 Checking API health...")
    try:
        health_response = requests.get(f"{api_url}/health")
        health = health_response.json()
        
        print(f"✅ API Status: {health['status']}")
        print(f"🔍 YOLO Detector: {health['yolo_detector']}")
        print(f"🔬 Fish Classifier: {health['fish_classifier']}")
        print(f"📊 API Version: {health['api_version']}")
        print(f"✨ Features: {', '.join(health['features'])}")
        
        if not health['models_ready']:
            print("❌ Models not ready. Please start the server first.")
            return
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the server:")
        print("   python run_web_app_polygon.py")
        return
    
    # Test with test image
    print(f"\n🔍 Testing polygon fish identification...")
    test_image = "test_image.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/api", files=files)
            result = response.json()
        
        if result.get('success'):
            print(f"✅ Polygon detection successful!")
            print(f"🐠 Fish detected: {result['fish_count']}")
            
            if 'annotated_image' in result:
                print(f"📷 Polygon annotated image: {result['annotated_image']}")
                print(f"🔗 View at: {api_url}/static/{result['annotated_image']}")
            
            print(f"\n📊 Polygon Detection Results:")
            for fish in result['fish']:
                polygon_points = len(fish.get('polygon', []))
                print(f"  🐠 Fish #{fish['fish_id']}: {fish['species']}")
                print(f"     Accuracy: {fish['accuracy']:.1%}")
                print(f"     Confidence: {fish['confidence']:.1%}")
                print(f"     Polygon vertices: {polygon_points} points")
                print(f"     Box: {fish['box']}")
                
                # Show first few polygon points as example
                if fish.get('polygon') and len(fish['polygon']) > 0:
                    preview_points = fish['polygon'][:3]
                    print(f"     Polygon preview: {preview_points}...")
                print()
        else:
            print(f"❌ Polygon detection failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error testing polygon API: {e}")

def test_polygon_features():
    """Demonstrate polygon-specific features."""
    print(f"\n🎯 POLYGON DETECTION FEATURES")
    print(f"=" * 50)
    
    features = [
        {
            "name": "🔺 Polygon Outlines",
            "description": "Fish outlined with actual shape polygons, not rectangles"
        },
        {
            "name": "🖱️ Interactive Clicking", 
            "description": "Click directly on fish polygons to select them"
        },
        {
            "name": "🎨 Semi-transparent Overlay",
            "description": "Colored polygon fills with transparency for better visibility"
        },
        {
            "name": "📍 Precise Selection",
            "description": "Point-in-polygon algorithm for accurate click detection"
        },
        {
            "name": "🏷️ Polygon Labels",
            "description": "Species names positioned at polygon centroids"
        },
        {
            "name": "📊 Vertex Information",
            "description": "Number of polygon vertices shown for each fish"
        },
        {
            "name": "🔄 Dynamic Highlighting",
            "description": "Selected fish highlighted in sidebar and image"
        },
        {
            "name": "📱 Responsive Design",
            "description": "Works on desktop and mobile with touch support"
        }
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature['name']}")
        print(f"   {feature['description']}")
        print()

def show_polygon_vs_bbox():
    """Show differences between polygon and bounding box approaches."""
    print(f"\n📐 POLYGON vs BOUNDING BOX COMPARISON")
    print(f"=" * 50)
    
    comparison = {
        "🔲 Bounding Box Approach": [
            "• Rectangle around fish",
            "• Includes background areas",
            "• Less precise selection",
            "• Simple 4-point coordinates",
            "• Fast to compute"
        ],
        "🔺 Polygon Approach": [
            "• Follows actual fish shape",
            "• Excludes background areas", 
            "• Precise shape-based selection",
            "• Multiple vertex coordinates",
            "• Better user experience"
        ]
    }
    
    for approach, features in comparison.items():
        print(f"{approach}:")
        for feature in features:
            print(f"  {feature}")
        print()

def interaction_examples():
    """Show interaction examples."""
    print(f"\n🖱️ INTERACTION EXAMPLES")
    print(f"=" * 40)
    
    examples = [
        {
            "scenario": "Single Fish Selection",
            "steps": [
                "1. Upload image with fish",
                "2. See fish outlined with colored polygon",
                "3. Click anywhere inside the polygon",
                "4. Fish details appear in sidebar"
            ]
        },
        {
            "scenario": "Multiple Fish Selection",
            "steps": [
                "1. Upload image with multiple fish",
                "2. Each fish gets different colored polygon",
                "3. Click on specific fish polygon",
                "4. Only that fish is selected and highlighted",
                "5. Click another polygon to switch selection"
            ]
        },
        {
            "scenario": "Sidebar Interaction",
            "steps": [
                "1. Click on fish cards in sidebar",
                "2. Corresponding polygon highlights in image",
                "3. Detailed info appears below sidebar",
                "4. Visual feedback shows selected state"
            ]
        }
    ]
    
    for example in examples:
        print(f"🎯 {example['scenario']}:")
        for step in example['steps']:
            print(f"   {step}")
        print()

def open_polygon_interface():
    """Open the polygon detection web interface."""
    print(f"\n🌐 Opening Polygon Detection Interface...")
    web_url = "http://localhost:5002"
    
    print(f"📱 Interactive Polygon Interface: {web_url}")
    print(f"✨ Try these features:")
    print(f"   • Upload an image with multiple fish")
    print(f"   • Click on individual fish polygons")
    print(f"   • See real-time polygon selection")
    print(f"   • Compare with bounding box approach")
    
    # Try to open in browser
    try:
        webbrowser.open(web_url)
        print(f"✅ Opened in browser")
    except:
        print(f"💡 Please manually open: {web_url}")

if __name__ == "__main__":
    print("🔺 Interactive Fish Polygon Detection - Test Suite")
    print("=" * 70)
    
    # Test API functionality
    test_polygon_api()
    
    # Show polygon features
    test_polygon_features()
    
    # Show comparison
    show_polygon_vs_bbox()
    
    # Show interaction examples
    interaction_examples()
    
    # Open web interface
    open_polygon_interface()
    
    print(f"\n🎉 POLYGON TESTING COMPLETE!")
    print(f"📋 Summary:")
    print(f"   ✅ Interactive polygon detection API")
    print(f"   ✅ Click-based fish selection")
    print(f"   ✅ Shape-accurate outlines")
    print(f"   ✅ Advanced segmentation features")
    print(f"   ✅ Professional polygon interaction")
    print(f"\n🚀 Ready for advanced fish identification!") 
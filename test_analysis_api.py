#!/usr/bin/env python3
"""
Fish Analysis API Test Client
============================

Test and demonstrate the Fish Analysis API functionality.
"""

import os
import json
import time
import requests
from pathlib import Path

class FishAnalysisAPIClient:
    """Client for testing Fish Analysis API."""
    
    def __init__(self, base_url="http://localhost:5004"):
        """Initialize API client."""
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test API health check."""
        print("🩺 Testing Health Check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Health Check: API is healthy")
                print(f"   Status: {data.get('status')}")
                print(f"   YOLOv12: {data.get('models', {}).get('yolo_v12', 'unknown')}")
                print(f"   Classifier: {data.get('models', {}).get('classifier', 'unknown')}")
                return True
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health Check Error: {e}")
            return False
    
    def test_model_info(self):
        """Test model information endpoint."""
        print("\n📊 Testing Model Information...")
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                print("✅ Model Info Retrieved")
                
                # Detection model
                detection = data.get('detection_model', {})
                if isinstance(detection, dict):
                    print(f"   Detection Model: {detection.get('model_type')} ({detection.get('species_count')} species)")
                else:
                    print(f"   Detection Model: {detection}")
                
                # Classification model
                classification = data.get('classification_model', {})
                print(f"   Classification Model: {classification.get('model_type')} v{classification.get('version')} ({classification.get('species_count')} species)")
                
                return True
            else:
                print(f"❌ Model Info Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model Info Error: {e}")
            return False
    
    def test_api_info(self):
        """Test API information endpoint."""
        print("\n📚 Testing API Information...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print("✅ API Info Retrieved")
                print(f"   Service: {data.get('service')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Description: {data.get('description')}")
                print(f"   Supported Formats: {', '.join(data.get('usage', {}).get('supported_formats', []))}")
                return True
            else:
                print(f"❌ API Info Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Info Error: {e}")
            return False
    
    def analyze_image(self, image_path, verbose=True):
        """Analyze fish image using API."""
        if verbose:
            print(f"\n🔍 Analyzing Image: {image_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"❌ Image file not found: {image_path}")
                return None
            
            # Prepare file upload
            with open(image_path, 'rb') as file:
                files = {'image': file}
                
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/analyze", files=files)
                end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if verbose:
                    print("✅ Analysis Complete!")
                    print(f"   API Response Time: {(end_time - start_time) * 1000:.2f}ms")
                    
                    if data.get('success'):
                        print(f"   Fish Count: {data.get('fish_count', 0)}")
                        print(f"   Processing Time: {data.get('processing_time', {}).get('total_ms', 0)}ms")
                        
                        # Show each fish
                        for fish in data.get('detections', []):
                            fish_id = fish.get('fish_id', 'Unknown')
                            detection = fish.get('detection', {})
                            classification = fish.get('classification', {})
                            metrics = fish.get('metrics', {})
                            
                            print(f"\n   🐟 Fish #{fish_id}:")
                            print(f"      YOLOv12: {detection.get('detected_class')} ({detection.get('confidence', 0):.3f})")
                            print(f"      Species: {classification.get('species')} ({classification.get('accuracy', 0):.3f})")
                            print(f"      Size: {metrics.get('size_category', 'Unknown')}")
                            print(f"      Position: {metrics.get('position', {})}")
                        
                        # Annotated image
                        annotated = data.get('annotated_image', {})
                        if annotated:
                            print(f"   📷 Annotated Image: {annotated.get('url')}")
                    else:
                        print(f"   ❌ Analysis Failed: {data.get('error')}")
                
                return data
            
            else:
                if verbose:
                    print(f"❌ Analysis Failed: HTTP {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data.get('error', 'Unknown error')}")
                    except:
                        print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            if verbose:
                print(f"❌ Analysis Error: {e}")
            return None
    
    def test_invalid_requests(self):
        """Test API with invalid requests."""
        print("\n🚫 Testing Invalid Requests...")
        
        # Test 1: No file
        try:
            response = self.session.post(f"{self.base_url}/analyze")
            if response.status_code == 400:
                print("✅ Correctly rejected request with no file")
            else:
                print(f"❌ Unexpected response for no file: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing no file: {e}")
        
        # Test 2: Invalid file type
        try:
            # Create temporary text file
            temp_file = "temp_test.txt"
            with open(temp_file, 'w') as f:
                f.write("This is not an image")
            
            with open(temp_file, 'rb') as file:
                files = {'image': file}
                response = self.session.post(f"{self.base_url}/analyze", files=files)
            
            os.remove(temp_file)
            
            if response.status_code == 400:
                print("✅ Correctly rejected invalid file type")
            else:
                print(f"❌ Unexpected response for invalid file: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing invalid file: {e}")
    
    def performance_test(self, image_path, iterations=3):
        """Test API performance with multiple requests."""
        print(f"\n⚡ Performance Test ({iterations} iterations)...")
        
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return
        
        times = []
        successful = 0
        
        for i in range(iterations):
            print(f"   Test {i+1}/{iterations}...", end=" ")
            
            start_time = time.time()
            result = self.analyze_image(image_path, verbose=False)
            end_time = time.time()
            
            if result and result.get('success'):
                duration = end_time - start_time
                times.append(duration)
                successful += 1
                fish_count = result.get('fish_count', 0)
                processing_time = result.get('processing_time', {}).get('total_ms', 0)
                print(f"✅ {duration*1000:.0f}ms (Found {fish_count} fish, Processed in {processing_time}ms)")
            else:
                print("❌ Failed")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n   📊 Performance Summary:")
            print(f"      Successful: {successful}/{iterations}")
            print(f"      Average Time: {avg_time*1000:.0f}ms")
            print(f"      Min Time: {min_time*1000:.0f}ms")
            print(f"      Max Time: {max_time*1000:.0f}ms")
        else:
            print("   ❌ No successful requests")

def main():
    """Run comprehensive API tests."""
    print("🐟 Fish Analysis API Test Suite")
    print("=" * 50)
    
    # Initialize client
    client = FishAnalysisAPIClient()
    
    # Test 1: Health check
    health_ok = client.test_health_check()
    
    # Test 2: Model information
    model_ok = client.test_model_info()
    
    # Test 3: API information
    api_ok = client.test_api_info()
    
    # Test 4: Image analysis
    test_image = "test_image.png"
    if os.path.exists(test_image):
        analysis_result = client.analyze_image(test_image)
        analysis_ok = analysis_result is not None and analysis_result.get('success', False)
    else:
        print(f"\n⚠️  Test image not found: {test_image}")
        print("   Skipping image analysis test")
        analysis_ok = None
    
    # Test 5: Invalid requests
    client.test_invalid_requests()
    
    # Test 6: Performance test
    if os.path.exists(test_image):
        client.performance_test(test_image, iterations=3)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Health Check: {'✅' if health_ok else '❌'}")
    print(f"   Model Info: {'✅' if model_ok else '❌'}")
    print(f"   API Info: {'✅' if api_ok else '❌'}")
    print(f"   Image Analysis: {'✅' if analysis_ok else '❌' if analysis_ok is False else '⚠️ Skipped'}")
    
    if all([health_ok, model_ok, api_ok]) and (analysis_ok or analysis_ok is None):
        print("\n🎉 All available tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the API server.")
    
    print("\n📝 Example Usage:")
    print("   curl -X POST -F 'image=@test_image.png' http://localhost:5004/analyze")
    print("   curl http://localhost:5004/health")
    print("   curl http://localhost:5004/models")

if __name__ == "__main__":
    main() 
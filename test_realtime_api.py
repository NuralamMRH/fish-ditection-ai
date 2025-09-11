#!/usr/bin/env python3
"""
Test Script for Real-time Fish Detection API
===========================================

This script demonstrates how to use the API endpoints programmatically.
"""

import requests
import json
import time
import os

# API Configuration
API_BASE_URL = "http://localhost:5009"
CALLBACK_URL = "https://www.itrucksea.com/fishing-log/batch"
TEST_IMAGE_PATH = "./test_image.png"  # Update with your test image path

def test_api_health():
    """Test API health check."""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"📊 Models: {data['models']['detectors']} detectors, Classifier: {data['models']['classifier']}")
        return True
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")
        return False

def test_fish_detection(image_path, callback_url=None):
    """Test fish detection endpoint."""
    print(f"\n🔍 Testing Fish Detection with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        files = {'image': open(image_path, 'rb')}
        
        # Test query parameter method (new preferred method)
        url = f"{API_BASE_URL}/api/detect"
        if callback_url:
            url += f"?callback_url={callback_url}"
            print(f"🔗 Using callback URL via query parameter: {callback_url}")
        
        response = requests.post(url, files=files)
        result = response.json()
        
        if result['success']:
            print(f"✅ Detection Success!")
            print(f"📊 Session ID: {result['session_id']}")
            print(f"🐟 Fish Count: {result['fish_count']}")
            print(f"🤖 Models Used: {result.get('models_used', 'N/A')}")
            print(f"🔗 Callback URL Set: {result.get('callback_url', 'None')}")
            
            if result['fish']:
                print("\n🐠 Fish Details:")
                for i, fish in enumerate(result['fish']):
                    weight = fish.get('weight', {}).get('weight_kg', 0)
                    length = fish.get('dimensions', {}).get('total_length_cm', 0)
                    print(f"  Fish {i+1}: {fish['species']} - {weight:.3f}kg, {length:.1f}cm")
            
            return result
        else:
            print(f"❌ Detection Failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Detection Error: {e}")
        return None

def test_get_results(session_id):
    """Test getting results for a session."""
    print(f"\n📋 Testing Get Results for session: {session_id}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/results/{session_id}")
        result = response.json()
        
        if 'result' in result:
            print(f"✅ Results Retrieved!")
            print(f"🐟 Fish Count: {result['result']['fish_count']}")
            print(f"📅 Created: {result['created_at']}")
            return result
        else:
            print(f"❌ Failed to get results: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Get Results Error: {e}")
        return None

def test_set_callback(session_id, callback_url):
    """Test setting callback URL for a session."""
    print(f"\n🔗 Testing Set Callback URL for session: {session_id}")
    
    try:
        data = {
            'session_id': session_id,
            'callback_url': callback_url
        }
        
        response = requests.post(f"{API_BASE_URL}/api/callback", json=data)
        result = response.json()
        
        if result['success']:
            print(f"✅ Callback URL Set!")
            print(f"🔗 URL: {result['callback_url']}")
            print(f"📤 Submit URL: {result['submit_url']}")
            return result
        else:
            print(f"❌ Failed to set callback: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Set Callback Error: {e}")
        return None

def test_submit_to_callback(session_id):
    """Test submitting results to callback URL."""
    print(f"\n📤 Testing Submit to Callback for session: {session_id}")
    
    try:
        # Add comprehensive test user info
        data = {
            'user_info': 'API Test User',
            'location': 'Test Lake',
            'fishing_method': 'API Testing',
            'weather': 'Sunny',
            'test_mode': True  # This prevents actual redirect in some cases
        }
        
        # Test API-only submission first
        response = requests.post(f"{API_BASE_URL}/api/submit/{session_id}?format=json", json=data)
        result = response.json()
        
        if result['success']:
            print(f"✅ Submission Success!")
            print(f"🔗 Callback URL: {result['callback_url']}")
            print(f"📋 Callback Submitted: {result['callback_submitted']}")
            print(f"🐟 Fish Count: {result['fish_count']}")
            
            if result['callback_response']:
                print(f"📊 Callback Status: {result['callback_response']['status_code']}")
                print(f"📝 Response: {result['callback_response']['response_text'][:100]}...")
            
            if result['callback_error']:
                print(f"⚠️ Callback Error: {result['callback_error']}")
            
            return result
        else:
            print(f"❌ Submission Failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Submit Error: {e}")
        return None

def test_api_status():
    """Test API status endpoint."""
    print(f"\n📊 Testing API Status...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/status")
        result = response.json()
        
        print(f"✅ API Status Retrieved!")
        print(f"📈 Active Sessions: {result['active_sessions']}")
        print(f"🎥 Camera Active: {result['camera_active']}")
        print(f"🔍 Detection Count: {result['detection_count']}")
        print(f"🤖 Models Available: {result['models_available']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Status Error: {e}")
        return None

def run_complete_test():
    """Run complete API test workflow."""
    print("🚀 Starting Complete API Test Workflow")
    print("=" * 50)
    
    # 1. Health Check
    if not test_api_health():
        print("❌ API not healthy, stopping tests")
        return
    
    # 2. Status Check
    test_api_status()
    
    # 3. Fish Detection (with callback)
    detection_result = test_fish_detection(TEST_IMAGE_PATH, CALLBACK_URL)
    if not detection_result:
        print("❌ Detection failed, stopping tests")
        return
    
    session_id = detection_result['session_id']
    
    # 4. Get Results
    test_get_results(session_id)
    
    # 5. Set Callback (if not set during detection)
    if not detection_result.get('callback_url'):
        test_set_callback(session_id, CALLBACK_URL)
    
    # 6. Submit to Callback
    test_submit_to_callback(session_id)
    
    print("\n✅ Complete API Test Workflow Finished!")
    print("=" * 50)

def demo_query_parameter_workflow():
    """Demonstrate the new query parameter callback workflow."""
    print("🔗 Demonstrating Query Parameter Callback Workflow")
    print("=" * 50)
    
    # Step 1: Upload image with callback URL via query parameter
    print("Step 1: Upload image with callback URL via query parameter")
    
    # Test different methods of setting callback URL
    print("\n🧪 Testing different callback URL methods:")
    
    # Method 1: Query parameter (new preferred method)
    print("\n1. Query Parameter Method:")
    detection_result = test_fish_detection(TEST_IMAGE_PATH, CALLBACK_URL)
    
    if detection_result and detection_result['fish_count'] > 0:
        session_id = detection_result['session_id']
        
        # Step 2: Review results
        print("\nStep 2: Review detection results")
        print(f"🎣 Detected {detection_result['fish_count']} fish")
        
        total_weight = sum(fish.get('weight', {}).get('weight_kg', 0) for fish in detection_result['fish'])
        print(f"⚖️ Total Weight: {total_weight:.3f} kg")
        
        # Step 3: Test REST API submission
        print("\nStep 3: Test REST API submission")
        submit_result = test_submit_to_callback(session_id)
        
        if submit_result:
            print(f"\n🚀 Query Parameter Workflow Complete!")
            print(f"📊 Fish data would be submitted to: {submit_result['callback_url']}")
            print(f"🔄 In browser, user would be redirected to: {submit_result['callback_url']}")
            
            # Show what data was sent to callback
            print(f"\n📋 Data sent to callback URL includes:")
            print(f"   - Session ID: {submit_result['session_id']}")
            print(f"   - Fish Count: {submit_result['fish_count']}")
            print(f"   - Callback Status: {submit_result['callback_submitted']}")
            
    print("=" * 50)

def test_callback_data_format():
    """Test and display the callback data format."""
    print("\n📋 Testing Callback Data Format")
    print("-" * 30)
    
    # Detect fish first
    detection_result = test_fish_detection(TEST_IMAGE_PATH, CALLBACK_URL)
    
    if detection_result:
        session_id = detection_result['session_id']
        
        # Get the full session data to show what gets sent to callback
        try:
            response = requests.get(f"{API_BASE_URL}/api/results/{session_id}")
            session_data = response.json()
            
            print("📊 Sample callback data structure:")
            callback_data = {
                'session_id': session_id,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'fish_data': session_data['result'],
                'source': 'fish_detection_api',
                'version': '2.0',
                'api_endpoint': f"{API_BASE_URL}/api/results/{session_id}",
                'total_fish_count': session_data['result']['fish_count'],
                'detection_success': session_data['result']['success']
            }
            
            print(json.dumps(callback_data, indent=2, default=str)[:800] + "...")
            
        except Exception as e:
            print(f"❌ Error getting session data: {e}")

if __name__ == "__main__":
    print("🐟 Real-time Fish Detection API Test Suite")
    print("=" * 60)
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"⚠️ Test image not found: {TEST_IMAGE_PATH}")
        print("Please update TEST_IMAGE_PATH in the script or place a test image")
        print("You can still run health and status checks...")
        
        # Just run health and status checks
        test_api_health()
        test_api_status()
    else:
        print("Choose test mode:")
        print("1. Complete API Test")
        print("2. Query Parameter Callback Demo")
        print("3. Callback Data Format Test")
        print("4. Health Check Only")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                run_complete_test()
            elif choice == "2":
                demo_query_parameter_workflow()
            elif choice == "3":
                test_callback_data_format()
            elif choice == "4":
                test_api_health()
                test_api_status()
            else:
                print("Invalid choice, running query parameter demo...")
                demo_query_parameter_workflow()
                
        except KeyboardInterrupt:
            print("\n\n🛑 Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test error: {e}")
    
    print("\n🏁 Test completed!") 
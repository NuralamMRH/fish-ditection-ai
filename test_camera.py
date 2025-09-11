#!/usr/bin/env python3
"""
Camera Test Script
=================

Simple script to test camera access and basic functionality.
"""

import cv2
import sys
import time

def test_camera(camera_id=0):
    """Test camera access and basic functionality."""
    print(f"🎥 Testing camera {camera_id}...")
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera {camera_id} opened successfully")
    print(f"📹 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    
    # Test frame capture
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        cap.release()
        return False
    
    print(f"✅ Frame captured: {frame.shape}")
    
    # Display test window
    print("🎥 Starting camera preview...")
    print("📋 Controls:")
    print("   Q - Quit")
    print("   S - Save screenshot")
    print("   ESC - Exit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Add frame info overlay
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add overlay text
            cv2.putText(frame, f"Camera {camera_id} Test", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow(f'Camera {camera_id} Test', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Screenshot
                filename = f"camera_{camera_id}_test_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Test Summary:")
        print(f"   Camera ID: {camera_id}")
        print(f"   Frames captured: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average FPS: {current_fps:.1f}")
        print("✅ Camera test completed")
    
    return True

def find_available_cameras():
    """Find all available cameras."""
    print("🔍 Scanning for available cameras...")
    available_cameras = []
    
    for i in range(10):  # Check cameras 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"✅ Camera {i} available")
            cap.release()
        else:
            print(f"❌ Camera {i} not available")
    
    if not available_cameras:
        print("❌ No cameras found!")
        return []
    
    print(f"📹 Found {len(available_cameras)} camera(s): {available_cameras}")
    return available_cameras

def main():
    """Main function."""
    print("🎥 Camera Test Utility")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
            print(f"📹 Testing specified camera: {camera_id}")
            success = test_camera(camera_id)
            if not success:
                print("💡 Try a different camera ID")
        except ValueError:
            if sys.argv[1] == "scan":
                find_available_cameras()
            else:
                print("❌ Invalid camera ID")
                print("💡 Usage: python test_camera.py [camera_id]")
                print("💡 Or: python test_camera.py scan")
    else:
        # Auto-detect and test cameras
        available_cameras = find_available_cameras()
        
        if available_cameras:
            print(f"\n🎥 Testing first available camera: {available_cameras[0]}")
            test_camera(available_cameras[0])
        else:
            print("❌ No cameras available for testing")
            print("💡 Make sure your camera is connected and not being used by another application")

if __name__ == "__main__":
    main() 
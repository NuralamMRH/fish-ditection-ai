#!/usr/bin/env python3
"""
Camera Permissions Setup
========================

Help users set up camera permissions and troubleshoot camera access issues.
"""

import os
import sys
import subprocess
import platform

def check_macos_permissions():
    """Check macOS camera permissions."""
    print("🔒 macOS Camera Permissions Check")
    print("=" * 50)
    
    try:
        # Try to get camera permission status
        result = subprocess.run([
            'sqlite3', 
            '/Users/$(whoami)/Library/Application Support/com.apple.TCC/TCC.db',
            "SELECT client,auth_value FROM access WHERE service='kTCCServiceCamera';"
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and result.stdout:
            print("📋 Current camera permissions:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    app, permission = line.split('|')
                    status = "✅ Allowed" if permission == '2' else "❌ Denied"
                    print(f"   {app}: {status}")
        
    except Exception as e:
        print(f"⚠️  Could not check permissions automatically: {e}")
    
    print("\n🔧 To enable camera access:")
    print("1. Go to: Apple Menu > System Preferences > Security & Privacy")
    print("2. Click on 'Privacy' tab")
    print("3. Select 'Camera' from the left sidebar")
    print("4. Make sure 'Terminal' or 'Python' is checked")
    print("5. If not listed, click '+' and add your Terminal app")
    print("6. Restart Terminal after making changes")
    
    print("\n💡 Alternative: Run Python from Applications folder:")
    print("1. Open Finder")
    print("2. Go to Applications")
    print("3. Find Python (or create alias)")
    print("4. Run camera scripts from there")

def check_camera_in_use():
    """Check if camera is being used by another process."""
    print("\n🔍 Checking for processes using camera...")
    
    try:
        # Check for common camera-using applications
        camera_apps = [
            'zoom', 'facetime', 'skype', 'teams', 'webex', 
            'obs', 'photobooth', 'quicktime', 'discord'
        ]
        
        running_camera_apps = []
        
        for app in camera_apps:
            result = subprocess.run(['pgrep', '-i', app], capture_output=True)
            if result.returncode == 0:
                running_camera_apps.append(app)
        
        if running_camera_apps:
            print("⚠️  Found applications that might be using camera:")
            for app in running_camera_apps:
                print(f"   • {app.title()}")
            print("\n💡 Try closing these applications and test again")
        else:
            print("✅ No obvious camera-using applications found")
            
    except Exception as e:
        print(f"⚠️  Could not check running processes: {e}")

def suggest_alternatives():
    """Suggest alternative testing methods."""
    print("\n🎯 Alternative Testing Options")
    print("=" * 50)
    
    print("1. 📱 Use Phone as IP Camera:")
    print("   - Install 'DroidCam' or similar app")
    print("   - Connect phone as network camera")
    print("   - Use with: python live_camera_detection.py 1")
    
    print("\n2. 📹 Use External USB Camera:")
    print("   - Connect USB webcam")
    print("   - Test with: python test_camera.py scan")
    
    print("\n3. 🎬 Use Video File for Testing:")
    print("   - Create video test script")
    print("   - Process recorded fish videos")
    
    print("\n4. 🖼️  Use Static Image Testing:")
    print("   - Test with existing images")
    print("   - Use: python simple_demo.py")

def create_video_test_script():
    """Create a script to test with video files."""
    video_test_code = '''#!/usr/bin/env python3
"""
Video File Fish Detection Test
============================

Test fish detection using video files instead of live camera.
"""

import cv2
import sys
import os

# Add detector_v12 to path
sys.path.insert(0, './detector_v12')
from local_inference import LocalYOLOv12Fish

def test_video_detection(video_path):
    """Test fish detection on video file."""
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Initialize detector
    try:
        detector = LocalYOLOv12Fish(confidence=0.4)
        print("✅ YOLOv12 model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    print(f"🎬 Processing video: {video_path}")
    frame_count = 0
    fish_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect fish
            detections = detector.predict(frame)
            
            if detections and detections[0]:
                fish_count = len(detections[0])
                fish_detections += fish_count
                
                # Draw detections
                for i, fish in enumerate(detections[0]):
                    box = fish.get_box()
                    confidence = fish.get_score()
                    fish_class = fish.get_class_name()
                    
                    x1, y1, x2, y2 = map(int, box)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{fish_class} ({confidence:.2f})", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"Frame {frame_count}: {fish_count} fish detected")
            
            # Display frame
            cv2.imshow('Video Fish Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\\n⏹️  Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\\n📊 Video Analysis Complete:")
        print(f"   Total frames: {frame_count}")
        print(f"   Fish detections: {fish_detections}")
        print(f"   Detection rate: {fish_detections/frame_count*100:.1f}%")

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_test.py <video_file>")
        print("Example: python video_test.py fish_video.mp4")
        return
    
    video_path = sys.argv[1]
    test_video_detection(video_path)

if __name__ == "__main__":
    main()
'''
    
    with open('video_test.py', 'w') as f:
        f.write(video_test_code)
    
    print("\n✅ Created video_test.py for testing with video files")
    print("Usage: python video_test.py your_fish_video.mp4")

def main():
    """Main setup function."""
    print("🐟 Fish Detection Camera Setup")
    print("=" * 60)
    
    # Detect operating system
    os_name = platform.system()
    print(f"🖥️  Operating System: {os_name}")
    
    if os_name == "Darwin":  # macOS
        check_macos_permissions()
        check_camera_in_use()
    elif os_name == "Linux":
        print("🐧 Linux detected")
        print("💡 Try: sudo usermod -a -G video $USER")
        print("   Then logout and login again")
    elif os_name == "Windows":
        print("🪟 Windows detected")
        print("💡 Check Windows Camera Privacy Settings")
        print("   Settings > Privacy > Camera")
    
    suggest_alternatives()
    create_video_test_script()
    
    print("\n🚀 Next Steps:")
    print("=" * 30)
    print("1. Fix camera permissions (see above)")
    print("2. Test camera: python test_camera.py")
    print("3. If camera works: python live_camera_detection.py")
    print("4. For enhanced features: python live_camera_enhanced.py")
    print("5. If no camera: Use video_test.py with video files")
    
    print("\n📚 Documentation:")
    print("   - Read LIVE_CAMERA_GUIDE.md for complete instructions")
    print("   - Check API_DOCUMENTATION.md for REST API usage")
    
    print("\n💡 Tips:")
    print("   - Start with static image testing first")
    print("   - Use test_image.png to verify model works")
    print("   - Camera permissions may require Terminal restart")

if __name__ == "__main__":
    main() 
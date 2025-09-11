#!/usr/bin/env python3
"""
Live Camera Fish Detection
=========================

Real-time fish detection using webcam and YOLOv12 model.
"""

import cv2
import sys
import time
import numpy as np
from datetime import datetime
import threading
import queue

# Add detector_v12 to path
sys.path.insert(0, './detector_v12')
from local_inference import LocalYOLOv12Fish

class LiveFishDetector:
    """Real-time fish detection from camera feed."""
    
    def __init__(self, camera_id=0, confidence=0.4):
        """Initialize live fish detector."""
        self.camera_id = camera_id
        self.confidence = confidence
        self.running = False
        
        # Initialize YOLOv12 detector
        try:
            self.detector = LocalYOLOv12Fish(confidence=confidence)
            print("✅ YOLOv12 model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLOv12: {e}")
            sys.exit(1)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            sys.exit(1)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Detection statistics
        self.total_detections = 0
        self.session_start_time = time.time()
        self.last_detection_time = 0
        self.detection_history = []
        
        # Colors for different fish species
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("🎥 Camera initialized successfully")
        print(f"📹 Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    def detect_fish_in_frame(self, frame):
        """Detect fish in a single frame."""
        try:
            # Run YOLOv12 detection
            detections = self.detector.predict(frame)
            
            if detections and detections[0]:
                return detections[0]
            return []
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detection results on frame."""
        if not detections:
            return frame
        
        annotated_frame = frame.copy()
        current_time = time.time()
        
        for i, fish in enumerate(detections):
            # Get detection info
            box = fish.get_box()
            confidence = fish.get_score()
            fish_class = fish.get_class_name()
            
            x1, y1, x2, y2 = map(int, box)
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare labels
            fish_label = f"{fish_class}"
            conf_label = f"{confidence:.2f}"
            
            # Calculate label sizes
            label_size = cv2.getTextSize(fish_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            conf_size = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1-35), (x1 + max(label_size[0], conf_size[0]) + 10, y1), color, -1)
            
            # Draw labels
            cv2.putText(annotated_frame, fish_label, (x1+5, y1-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, conf_label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Update detection statistics
            self.total_detections += 1
            self.last_detection_time = current_time
            
            # Add to detection history (keep last 100)
            self.detection_history.append({
                'timestamp': current_time,
                'species': fish_class,
                'confidence': confidence
            })
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
        
        return annotated_frame
    
    def draw_ui_overlay(self, frame):
        """Draw UI overlay with statistics and controls."""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Background for statistics
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "🐟 Live Fish Detection", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Statistics
        session_time = time.time() - self.session_start_time
        time_since_last = time.time() - self.last_detection_time if self.last_detection_time > 0 else 0
        
        stats = [
            f"FPS: {self.current_fps:.1f}",
            f"Total Detections: {self.total_detections}",
            f"Session Time: {int(session_time//60):02d}:{int(session_time%60):02d}",
            f"Model: YOLOv12 ({len(self.detector.class_names)} species)"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recent detections
        if self.detection_history:
            cv2.putText(frame, "Recent Detections:", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show last 3 detections
            recent = self.detection_history[-3:] if len(self.detection_history) >= 3 else self.detection_history
            for i, detection in enumerate(recent):
                ago = time.time() - detection['timestamp']
                text = f"• {detection['species']} ({detection['confidence']:.2f}) - {ago:.1f}s ago"
                cv2.putText(frame, text, (25, 160 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Controls
        control_y = height - 60
        cv2.rectangle(frame, (10, control_y-10), (width-10, height-10), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        controls = [
            "Controls: [Q]uit | [S]creenshot | [R]eset Stats | [C]onfidence +/- | [F]ullscreen"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (20, control_y + 10 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def calculate_fps(self):
        """Calculate current FPS."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fish_detection_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.total_detections = 0
        self.session_start_time = time.time()
        self.last_detection_time = 0
        self.detection_history = []
        print("📊 Statistics reset")
    
    def run(self):
        """Main detection loop."""
        print("\n🚀 Starting live fish detection...")
        print("📋 Controls:")
        print("   Q - Quit")
        print("   S - Take screenshot")
        print("   R - Reset statistics")
        print("   + - Increase confidence threshold")
        print("   - - Decrease confidence threshold")
        print("   F - Toggle fullscreen")
        print("   ESC - Quit")
        print("\n🎥 Camera feed starting...")
        
        self.running = True
        fullscreen = False
        
        # Create window
        cv2.namedWindow('Live Fish Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Fish Detection', 800, 600)
        
        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect fish in frame
                detections = self.detect_fish_in_frame(frame)
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Draw UI overlay
                frame = self.draw_ui_overlay(frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display frame
                cv2.imshow('Live Fish Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):  # Screenshot
                    self.save_screenshot(frame)
                elif key == ord('r'):  # Reset stats
                    self.reset_statistics()
                elif key == ord('+') or key == ord('='):  # Increase confidence
                    self.confidence = min(0.9, self.confidence + 0.05)
                    self.detector.confidence = self.confidence
                    print(f"🎯 Confidence threshold: {self.confidence:.2f}")
                elif key == ord('-'):  # Decrease confidence
                    self.confidence = max(0.1, self.confidence - 0.05)
                    self.detector.confidence = self.confidence
                    print(f"🎯 Confidence threshold: {self.confidence:.2f}")
                elif key == ord('f'):  # Fullscreen toggle
                    if fullscreen:
                        cv2.setWindowProperty('Live Fish Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        fullscreen = False
                    else:
                        cv2.setWindowProperty('Live Fish Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        fullscreen = True
                        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        self.running = False
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Final statistics
        session_time = time.time() - self.session_start_time
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {int(session_time//60):02d}:{int(session_time%60):02d}")
        print(f"   Total Detections: {self.total_detections}")
        print(f"   Average Detection Rate: {self.total_detections/session_time*60:.1f} per minute")
        
        if self.detection_history:
            species_count = {}
            for det in self.detection_history:
                species = det['species']
                species_count[species] = species_count.get(species, 0) + 1
            
            print(f"   Species Detected:")
            for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True):
                print(f"     • {species}: {count} times")
        
        print("👋 Goodbye!")

def main():
    """Main function to run live fish detection."""
    print("🐟 Live Fish Detection System")
    print("=" * 50)
    
    # Check for camera argument
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print("⚠️  Invalid camera ID, using default (0)")
    
    # Check for confidence argument
    confidence = 0.4
    if len(sys.argv) > 2:
        try:
            confidence = float(sys.argv[2])
            confidence = max(0.1, min(0.9, confidence))
        except ValueError:
            print("⚠️  Invalid confidence value, using default (0.4)")
    
    print(f"📹 Using camera ID: {camera_id}")
    print(f"🎯 Confidence threshold: {confidence}")
    
    # Initialize and run detector
    try:
        detector = LiveFishDetector(camera_id=camera_id, confidence=confidence)
        detector.run()
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("💡 Try different camera ID: python live_camera_detection.py 1")

if __name__ == "__main__":
    main() 
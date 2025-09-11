#!/usr/bin/env python3
"""
Enhanced Live Camera Fish Detection
==================================

Advanced real-time fish detection with recording, multiple cameras, and enhanced features.
"""

import cv2
import sys
import time
import numpy as np
from datetime import datetime
import threading
import queue
import os
import json
import argparse

# Add detector_v12 to path
sys.path.insert(0, './detector_v12')
from local_inference import LocalYOLOv12Fish

class EnhancedLiveFishDetector:
    """Enhanced real-time fish detection with advanced features."""
    
    def __init__(self, camera_id=0, confidence=0.4, record_video=False, 
                 motion_detection=False, detection_zone=None):
        """Initialize enhanced live fish detector."""
        self.camera_id = camera_id
        self.confidence = confidence
        self.record_video = record_video
        self.motion_detection = motion_detection
        self.detection_zone = detection_zone
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
        
        # Get camera properties
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set optimal camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video recording setup
        self.video_writer = None
        self.recording_start_time = None
        self.recorded_detections = 0
        
        # Motion detection setup
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_threshold = 500
        
        # Detection statistics
        self.total_detections = 0
        self.session_start_time = time.time()
        self.last_detection_time = 0
        self.detection_history = []
        self.hourly_stats = {}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processing_times = []
        
        # UI settings
        self.show_fps = True
        self.show_stats = True
        self.show_zones = True
        self.night_mode = False
        
        # Colors
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
        
        print("🎥 Enhanced camera initialized successfully")
        print(f"📹 Resolution: {self.frame_width}x{self.frame_height}")
        print(f"🔍 Motion Detection: {'Enabled' if motion_detection else 'Disabled'}")
        print(f"📽️  Video Recording: {'Enabled' if record_video else 'Disabled'}")
    
    def setup_recording(self):
        """Setup video recording."""
        if not self.record_video:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fish_detection_recording_{timestamp}.mp4"
        
        # Create recordings directory
        os.makedirs('recordings', exist_ok=True)
        filepath = os.path.join('recordings', filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            filepath, fourcc, 20.0, (self.frame_width, self.frame_height)
        )
        
        self.recording_start_time = time.time()
        self.recorded_detections = 0
        print(f"🎬 Recording started: {filename}")
    
    def detect_motion(self, frame):
        """Detect motion in frame."""
        if not self.motion_detection:
            return False
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Count non-zero pixels
        motion_pixels = cv2.countNonZero(fg_mask)
        
        return motion_pixels > self.motion_threshold
    
    def is_in_detection_zone(self, box):
        """Check if detection is within specified zone."""
        if not self.detection_zone:
            return True
        
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        zone_x1, zone_y1, zone_x2, zone_y2 = self.detection_zone
        
        return (zone_x1 <= center_x <= zone_x2 and 
                zone_y1 <= center_y <= zone_y2)
    
    def detect_fish_in_frame(self, frame):
        """Detect fish in frame with motion filtering."""
        start_time = time.time()
        
        try:
            # Motion detection filter
            if self.motion_detection:
                has_motion = self.detect_motion(frame)
                if not has_motion:
                    return [], time.time() - start_time
            
            # Run YOLOv12 detection
            detections = self.detector.predict(frame)
            
            if detections and detections[0]:
                # Filter by detection zone
                if self.detection_zone:
                    filtered_detections = []
                    for fish in detections[0]:
                        box = fish.get_box()
                        if self.is_in_detection_zone(box):
                            filtered_detections.append(fish)
                    return filtered_detections, time.time() - start_time
                else:
                    return detections[0], time.time() - start_time
            
            return [], time.time() - start_time
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], time.time() - start_time
    
    def draw_detection_zones(self, frame):
        """Draw detection zones on frame."""
        if not self.detection_zone or not self.show_zones:
            return frame
        
        zone_x1, zone_y1, zone_x2, zone_y2 = self.detection_zone
        
        # Draw zone rectangle
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), 
                     (0, 255, 255), 2)
        
        # Add zone label
        cv2.putText(frame, "Detection Zone", (zone_x1, zone_y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def draw_detections(self, frame, detections):
        """Draw detection results with enhanced visuals."""
        if not detections:
            return frame
        
        annotated_frame = frame.copy()
        current_time = time.time()
        current_hour = datetime.now().hour
        
        for i, fish in enumerate(detections):
            # Get detection info
            box = fish.get_box()
            confidence = fish.get_score()
            fish_class = fish.get_class_name()
            
            x1, y1, x2, y2 = map(int, box)
            color = self.colors[i % len(self.colors)]
            
            # Enhanced bounding box with glow effect
            cv2.rectangle(annotated_frame, (x1-2, y1-2), (x2+2, y2+2), color, 3)
            cv2.rectangle(annotated_frame, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)
            
            # Fish ID badge
            fish_id = f"#{i+1}"
            badge_size = cv2.getTextSize(fish_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.circle(annotated_frame, (x1-15, y1-15), 15, color, -1)
            cv2.putText(annotated_frame, fish_id, (x1-22, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Prepare labels
            fish_label = f"{fish_class}"
            conf_label = f"{confidence:.3f}"
            size_label = f"{x2-x1}x{y2-y1}px"
            
            # Enhanced label background
            max_width = max(
                cv2.getTextSize(fish_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0],
                cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0],
                cv2.getTextSize(size_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0]
            )
            
            cv2.rectangle(annotated_frame, (x1, y1-50), (x1 + max_width + 20, y1), color, -1)
            cv2.rectangle(annotated_frame, (x1, y1-50), (x1 + max_width + 20, y1), (255, 255, 255), 1)
            
            # Multi-line labels
            cv2.putText(annotated_frame, fish_label, (x1+5, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, conf_label, (x1+5, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, size_label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Update statistics
            self.total_detections += 1
            self.last_detection_time = current_time
            self.recorded_detections += 1
            
            # Hourly statistics
            if current_hour not in self.hourly_stats:
                self.hourly_stats[current_hour] = {'count': 0, 'species': {}}
            
            self.hourly_stats[current_hour]['count'] += 1
            species_dict = self.hourly_stats[current_hour]['species']
            species_dict[fish_class] = species_dict.get(fish_class, 0) + 1
            
            # Add to detection history
            self.detection_history.append({
                'timestamp': current_time,
                'species': fish_class,
                'confidence': confidence,
                'box': box,
                'size': (x2-x1) * (y2-y1)
            })
            
            if len(self.detection_history) > 200:
                self.detection_history.pop(0)
        
        return annotated_frame
    
    def draw_enhanced_ui(self, frame, processing_time):
        """Draw enhanced UI overlay."""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Adaptive UI colors based on night mode
        bg_color = (20, 20, 20) if self.night_mode else (0, 0, 0)
        text_color = (200, 200, 200) if self.night_mode else (255, 255, 255)
        accent_color = (0, 255, 255)
        
        # Main statistics panel
        if self.show_stats:
            panel_width = 400
            panel_height = 200
            cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), bg_color, -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (10, 10), (panel_width, panel_height), accent_color, 2)
            
            # Title with emoji
            cv2.putText(frame, "🐟 Enhanced Fish Detection", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent_color, 2)
            
            # Real-time statistics
            session_time = time.time() - self.session_start_time
            recording_time = time.time() - self.recording_start_time if self.recording_start_time else 0
            
            stats = [
                f"FPS: {self.current_fps:.1f}",
                f"Processing: {processing_time*1000:.1f}ms",
                f"Total Fish: {self.total_detections}",
                f"Session: {int(session_time//60):02d}:{int(session_time%60):02d}",
                f"Model: YOLOv12 ({len(self.detector.class_names)} species)",
                f"Recording: {'ON' if self.record_video else 'OFF'} - {int(recording_time//60):02d}:{int(recording_time%60):02d}"
            ]
            
            for i, stat in enumerate(stats):
                color = accent_color if i < 2 else text_color
                cv2.putText(frame, stat, (20, 60 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Recent detections panel
        if self.detection_history and self.show_stats:
            recent_panel_y = 220
            cv2.rectangle(overlay, (10, recent_panel_y), (400, recent_panel_y + 120), bg_color, -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (10, recent_panel_y), (400, recent_panel_y + 120), (0, 255, 0), 2)
            
            cv2.putText(frame, "📊 Recent Detections", (20, recent_panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show last 4 detections
            recent = self.detection_history[-4:] if len(self.detection_history) >= 4 else self.detection_history
            for i, detection in enumerate(recent):
                ago = time.time() - detection['timestamp']
                size = detection['size']
                text = f"• {detection['species']} ({detection['confidence']:.2f}) {size}px² - {ago:.1f}s"
                cv2.putText(frame, text, (20, recent_panel_y + 45 + i*17), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Performance indicators
        if self.show_fps:
            # FPS indicator
            fps_color = (0, 255, 0) if self.current_fps > 15 else (0, 255, 255) if self.current_fps > 10 else (0, 0, 255)
            cv2.circle(frame, (width - 50, 30), 20, fps_color, -1)
            cv2.putText(frame, f"{int(self.current_fps)}", (width - 60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Recording indicator
        if self.record_video:
            recording_color = (0, 0, 255) if int(time.time()) % 2 else (0, 100, 255)
            cv2.circle(frame, (width - 100, 30), 10, recording_color, -1)
            cv2.putText(frame, "REC", (width - 125, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, recording_color, 2)
        
        # Enhanced controls panel
        control_y = height - 100
        cv2.rectangle(overlay, (10, control_y), (width-10, height-10), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        controls = [
            "🎮 Controls: [Q]uit | [S]creenshot | [R]ecord | [SPACE]Reset | [T]oggle Stats | [N]ight Mode",
            "⚙️  Settings: [+/-]Confidence | [F]ullscreen | [M]otion Detection | [Z]one Toggle",
            "📊 Info: [I]nfo Panel | [H]ourly Stats | [ESC]Emergency Exit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (20, control_y + 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
        
        return frame
    
    def save_hourly_stats(self):
        """Save hourly statistics to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fish_detection_stats_{timestamp}.json"
        
        stats_data = {
            'session_start': datetime.fromtimestamp(self.session_start_time).isoformat(),
            'total_detections': self.total_detections,
            'hourly_breakdown': self.hourly_stats,
            'detection_history': self.detection_history[-50:],  # Last 50 detections
            'camera_id': self.camera_id,
            'confidence_threshold': self.confidence
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"📊 Statistics saved: {filename}")
    
    def calculate_fps(self):
        """Calculate and update FPS."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_screenshot(self, frame):
        """Save enhanced screenshot with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fish_detection_screenshot_{timestamp}.jpg"
        
        # Add metadata overlay
        metadata_frame = frame.copy()
        metadata_text = f"Fish Detection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Total: {self.total_detections}"
        cv2.putText(metadata_frame, metadata_text, (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(filename, metadata_frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def toggle_recording(self):
        """Toggle video recording."""
        if self.record_video and self.video_writer:
            # Stop recording
            self.video_writer.release()
            self.video_writer = None
            recording_time = time.time() - self.recording_start_time
            print(f"⏹️  Recording stopped. Duration: {int(recording_time//60):02d}:{int(recording_time%60):02d}")
            print(f"📊 Fish detected during recording: {self.recorded_detections}")
            self.record_video = False
        else:
            # Start recording
            self.record_video = True
            self.setup_recording()
    
    def run(self):
        """Main enhanced detection loop."""
        print("\n🚀 Starting Enhanced Live Fish Detection...")
        print("📋 Enhanced Controls:")
        print("   Q/ESC - Quit")
        print("   S - Screenshot")
        print("   R - Toggle Recording")
        print("   SPACE - Reset Statistics")
        print("   T - Toggle Statistics Panel")
        print("   N - Night Mode")
        print("   +/- - Adjust Confidence")
        print("   F - Fullscreen")
        print("   M - Toggle Motion Detection")
        print("   Z - Toggle Detection Zones")
        print("   I - Toggle Info Panel")
        print("   H - Save Hourly Statistics")
        print("\n🎥 Enhanced camera feed starting...")
        
        self.running = True
        fullscreen = False
        
        # Setup recording if enabled
        if self.record_video:
            self.setup_recording()
        
        # Create window
        cv2.namedWindow('Enhanced Fish Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced Fish Detection', 1280, 720)
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Apply detection zones
                frame = self.draw_detection_zones(frame)
                
                # Detect fish
                detections, processing_time = self.detect_fish_in_frame(frame)
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Draw enhanced UI
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                frame = self.draw_enhanced_ui(frame, avg_processing_time)
                
                # Record frame if recording
                if self.record_video and self.video_writer:
                    self.video_writer.write(frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display frame
                cv2.imshow('Enhanced Fish Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):  # Screenshot
                    self.save_screenshot(frame)
                elif key == ord('r'):  # Toggle recording
                    self.toggle_recording()
                elif key == ord(' '):  # Reset statistics
                    self.reset_statistics()
                elif key == ord('t'):  # Toggle stats
                    self.show_stats = not self.show_stats
                elif key == ord('n'):  # Night mode
                    self.night_mode = not self.night_mode
                elif key == ord('+') or key == ord('='):  # Increase confidence
                    self.confidence = min(0.9, self.confidence + 0.05)
                    self.detector.confidence = self.confidence
                    print(f"🎯 Confidence: {self.confidence:.2f}")
                elif key == ord('-'):  # Decrease confidence
                    self.confidence = max(0.1, self.confidence - 0.05)
                    self.detector.confidence = self.confidence
                    print(f"🎯 Confidence: {self.confidence:.2f}")
                elif key == ord('f'):  # Fullscreen
                    if fullscreen:
                        cv2.setWindowProperty('Enhanced Fish Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        fullscreen = False
                    else:
                        cv2.setWindowProperty('Enhanced Fish Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        fullscreen = True
                elif key == ord('m'):  # Motion detection
                    self.motion_detection = not self.motion_detection
                    print(f"🏃 Motion Detection: {'ON' if self.motion_detection else 'OFF'}")
                elif key == ord('z'):  # Toggle zones
                    self.show_zones = not self.show_zones
                elif key == ord('i'):  # Info panel
                    self.show_fps = not self.show_fps
                elif key == ord('h'):  # Save hourly stats
                    self.save_hourly_stats()
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_detections = 0
        self.session_start_time = time.time()
        self.last_detection_time = 0
        self.detection_history = []
        self.hourly_stats = {}
        self.recorded_detections = 0
        print("📊 All statistics reset")
    
    def cleanup(self):
        """Enhanced cleanup with detailed statistics."""
        print("\n🧹 Enhanced cleanup...")
        self.running = False
        
        # Stop recording
        if self.record_video and self.video_writer:
            self.video_writer.release()
            recording_time = time.time() - self.recording_start_time
            print(f"📹 Final recording: {int(recording_time//60):02d}:{int(recording_time%60):02d}")
        
        # Release camera
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Enhanced session summary
        session_time = time.time() - self.session_start_time
        avg_fps = self.current_fps
        avg_processing = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        print(f"\n📊 Enhanced Session Summary:")
        print(f"   Duration: {int(session_time//3600):02d}:{int((session_time%3600)//60):02d}:{int(session_time%60):02d}")
        print(f"   Total Detections: {self.total_detections}")
        print(f"   Detection Rate: {self.total_detections/session_time*60:.2f}/min")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Average Processing: {avg_processing*1000:.1f}ms")
        
        # Species breakdown
        if self.detection_history:
            species_count = {}
            for det in self.detection_history:
                species = det['species']
                species_count[species] = species_count.get(species, 0) + 1
            
            print(f"   Species Detected:")
            for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / len(self.detection_history)) * 100
                print(f"     • {species}: {count} ({percentage:.1f}%)")
        
        # Hourly breakdown
        if self.hourly_stats:
            print(f"   Hourly Activity:")
            for hour in sorted(self.hourly_stats.keys()):
                count = self.hourly_stats[hour]['count']
                print(f"     • {hour:02d}:00-{hour:02d}:59: {count} detections")
        
        print("👋 Enhanced Fish Detection Complete!")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced Live Fish Detection')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--confidence', '-conf', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    parser.add_argument('--record', '-r', action='store_true', help='Enable video recording')
    parser.add_argument('--motion', '-m', action='store_true', help='Enable motion detection')
    parser.add_argument('--zone', '-z', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'), 
                       help='Detection zone coordinates: x1 y1 x2 y2')
    
    args = parser.parse_args()
    
    print("🐟 Enhanced Live Fish Detection System")
    print("=" * 60)
    print(f"📹 Camera ID: {args.camera}")
    print(f"🎯 Confidence: {args.confidence}")
    print(f"📽️  Recording: {'Enabled' if args.record else 'Disabled'}")
    print(f"🏃 Motion Detection: {'Enabled' if args.motion else 'Disabled'}")
    
    if args.zone:
        print(f"🎯 Detection Zone: {args.zone}")
    
    try:
        detector = EnhancedLiveFishDetector(
            camera_id=args.camera,
            confidence=args.confidence,
            record_video=args.record,
            motion_detection=args.motion,
            detection_zone=args.zone
        )
        detector.run()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Try: python live_camera_enhanced.py --camera 1")

if __name__ == "__main__":
    main() 
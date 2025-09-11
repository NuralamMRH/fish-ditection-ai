# Live Camera Fish Detection Guide

## Overview

The Live Camera Fish Detection system provides real-time fish detection using your webcam or external camera. It uses your trained YOLOv12 model to detect and identify fish species in real-time with visual overlays, statistics tracking, and recording capabilities.

## Features

### 🎥 Basic Live Detection (`live_camera_detection.py`)

- Real-time fish detection from webcam
- Visual bounding boxes and species labels
- FPS monitoring and performance metrics
- Detection statistics and history
- Screenshot capture
- Adjustable confidence threshold
- Fullscreen mode

### 🚀 Enhanced Live Detection (`live_camera_enhanced.py`)

- All basic features plus:
- **Video Recording** - Record detection sessions with annotations
- **Motion Detection** - Only process frames with movement
- **Detection Zones** - Define specific areas for detection
- **Hourly Statistics** - Track detections by hour
- **Night Mode** - Dark UI for low-light environments
- **Multiple Camera Support** - Switch between cameras
- **Enhanced UI** - Professional overlay with detailed metrics
- **Data Export** - Save statistics to JSON files

## Setup Instructions

### 1. Prerequisites

Make sure you have the required dependencies:

```bash
# Activate virtual environment
source venv/bin/activate

# Verify OpenCV installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### 2. Test Camera Access

Before running fish detection, test your camera:

```bash
# Test default camera (usually built-in webcam)
python test_camera.py

# Test specific camera ID
python test_camera.py 1

# Scan for all available cameras
python test_camera.py scan
```

Expected output:

```
✅ Camera 0 available
📹 Found 1 camera(s): [0]
✅ Camera 0 opened successfully
📹 Resolution: 1280x720
🎬 FPS: 30.0
```

### 3. Camera Troubleshooting

If camera test fails:

**macOS:**

```bash
# Check camera permissions
# Go to: System Preferences > Security & Privacy > Camera
# Make sure Terminal/Python has camera access

# Check if camera is being used by another app
lsof | grep -i camera
```

**Common Issues:**

- Camera already in use by another application (Zoom, FaceTime, etc.)
- Missing camera permissions
- Wrong camera ID
- Camera driver issues

## Usage Examples

### Basic Live Detection

```bash
# Start with default camera (ID 0)
python live_camera_detection.py

# Use specific camera
python live_camera_detection.py 1 0.5
#                                ^ camera_id ^ confidence
```

**Basic Controls:**

- `Q` - Quit
- `S` - Take screenshot
- `R` - Reset statistics
- `+/-` - Adjust confidence threshold
- `F` - Toggle fullscreen
- `ESC` - Emergency exit

### Enhanced Live Detection

```bash
# Basic enhanced detection
python live_camera_enhanced.py

# With recording enabled
python live_camera_enhanced.py --record

# With motion detection
python live_camera_enhanced.py --motion

# With specific detection zone
python live_camera_enhanced.py --zone 100 100 500 400

# Full example with all features
python live_camera_enhanced.py \
  --camera 1 \
  --confidence 0.3 \
  --record \
  --motion \
  --zone 200 150 800 600
```

**Enhanced Controls:**

- `Q/ESC` - Quit
- `S` - Screenshot with metadata
- `R` - Toggle video recording
- `SPACE` - Reset all statistics
- `T` - Toggle statistics panel
- `N` - Night mode
- `+/-` - Adjust confidence threshold
- `F` - Fullscreen
- `M` - Toggle motion detection
- `Z` - Toggle detection zones
- `I` - Toggle info panels
- `H` - Save hourly statistics

## Configuration Options

### Camera Settings

```python
# In the script, you can modify these settings:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Resolution width
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Resolution height
self.cap.set(cv2.CAP_PROP_FPS, 30)             # Frame rate
self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Buffer size
```

### Detection Parameters

```bash
# Confidence threshold (0.1 - 0.9)
--confidence 0.4    # Default: detects most fish
--confidence 0.6    # More strict: fewer false positives
--confidence 0.2    # More sensitive: catches small/distant fish
```

### Detection Zones

```bash
# Define rectangular detection area
--zone x1 y1 x2 y2

# Examples:
--zone 0 0 640 480      # Full frame (640x480)
--zone 100 100 540 380  # Center area only
--zone 0 200 640 480    # Bottom half only
```

## Output Files

### Screenshots

- Format: `fish_detection_screenshot_YYYYMMDD_HHMMSS.jpg`
- Location: Current directory
- Includes: Timestamp and detection count overlay

### Video Recordings

- Format: `fish_detection_recording_YYYYMMDD_HHMMSS.mp4`
- Location: `./recordings/` directory
- Includes: All visual overlays and annotations
- Codec: MP4V (widely compatible)

### Statistics Files

- Format: `fish_detection_stats_YYYYMMDD_HHMMSS.json`
- Contains: Session data, hourly breakdown, detection history
- Example structure:

```json
{
  "session_start": "2025-06-25T12:00:00",
  "total_detections": 42,
  "hourly_breakdown": {
    "12": {"count": 15, "species": {"Tuna": 8, "Salmon": 7}},
    "13": {"count": 27, "species": {"Tuna": 12, "Bass": 15}}
  },
  "detection_history": [...],
  "camera_id": 0,
  "confidence_threshold": 0.4
}
```

## Performance Optimization

### Hardware Recommendations

- **CPU**: Modern multi-core processor (i5/i7 or equivalent)
- **RAM**: 8GB+ (YOLOv12 model uses ~2GB)
- **Camera**: USB 3.0 or built-in HD camera
- **Resolution**: 720p recommended, 1080p if CPU allows

### Performance Settings

```bash
# For better performance on slower machines:
python live_camera_enhanced.py --confidence 0.5  # Higher threshold = faster
# Modify resolution in code: 640x480 instead of 1280x720

# For better accuracy on powerful machines:
python live_camera_enhanced.py --confidence 0.3  # Lower threshold = more detections
# Use 1080p resolution for better detail
```

### Frame Rate Optimization

The system automatically adjusts to maintain real-time performance:

- **30+ FPS**: Excellent (real-time)
- **15-30 FPS**: Good (minor delay)
- **10-15 FPS**: Acceptable (noticeable delay)
- **<10 FPS**: Poor (consider lower resolution/higher confidence)

## Species Detection

Your YOLOv12 model can detect these 28 fish species:

- Anchovies
- Bangus
- Basa fish
- Big-Head-Carp
- Black-Spotted-Barb
- Blue marlin
- Catfish
- Climbing-Perch
- Cow tongue fish
- Crucian carp
- Fourfinger-Threadfin
- Freshwater-Eel
- Giant Grouper
- Glass-Perchlet
- Goby
- Gold-Fish
- Mackerel
- Mullet fish
- Northern red snapper
- Perch fish
- Phu Quoc Island Tuna
- Pompano
- Rabbitfish
- Snakehead fish
- Snapper
- Tuna
- Vietnamese mackerel
- big head carp

## Real-World Usage Tips

### Aquarium Monitoring

```bash
# Set up detection zone for tank area only
python live_camera_enhanced.py \
  --record \
  --zone 100 50 700 450 \
  --confidence 0.3
```

### Fish Farm Monitoring

```bash
# Use motion detection to save processing power
python live_camera_enhanced.py \
  --motion \
  --record \
  --confidence 0.4
```

### Research/Counting

```bash
# Enable all statistics tracking
python live_camera_enhanced.py \
  --record \
  --confidence 0.5
# Use H key to save hourly statistics
```

## Troubleshooting

### Common Issues

**1. "Failed to open camera"**

```bash
# Check available cameras
python test_camera.py scan

# Try different camera IDs
python live_camera_detection.py 1
python live_camera_detection.py 2
```

**2. "YOLOv12 model not found"**

```bash
# Verify model file exists
ls -la detector_v12/best.pt

# Re-run training if needed
cd detector_v12
python train_local.py
```

**3. Low FPS / Poor Performance**

```bash
# Lower resolution in code or use higher confidence
python live_camera_enhanced.py --confidence 0.6

# Close other applications using camera/CPU
```

**4. No Fish Detected**

```bash
# Lower confidence threshold
python live_camera_detection.py 0 0.2

# Check if objects look like fish to the model
# Try with test images first
```

### Debug Mode

Add debug prints to understand what's happening:

```python
# In detect_fish_in_frame method, add:
print(f"Detections found: {len(detections) if detections else 0}")
print(f"Processing time: {processing_time*1000:.1f}ms")
```

## Advanced Features

### Custom Detection Zones

You can define multiple zones by modifying the code:

```python
# Multiple detection zones
self.detection_zones = [
    (100, 100, 400, 300),  # Zone 1
    (500, 200, 800, 500),  # Zone 2
]
```

### Integration with Fish Analysis API

Combine live detection with the analysis API:

```python
# Save detected fish crops for detailed analysis
for fish in detections:
    crop = fish.get_mask_BGR()
    cv2.imwrite(f"detected_fish_{time.time()}.jpg", crop)
    # Send to analysis API for species classification
```

### Automated Recording

Set up automated recording based on conditions:

```python
# Auto-record when fish detected
if detections and not self.recording:
    self.toggle_recording()
```

## System Requirements

### Minimum Requirements

- Python 3.8+
- OpenCV 4.0+
- 4GB RAM
- USB 2.0 camera
- Dual-core CPU

### Recommended Requirements

- Python 3.9+
- OpenCV 4.5+
- 8GB+ RAM
- USB 3.0 HD camera
- Quad-core CPU
- Dedicated GPU (optional, for faster processing)

## Support

If you encounter issues:

1. **Test basic camera access** with `test_camera.py`
2. **Check model loading** - ensure `detector_v12/best.pt` exists
3. **Verify dependencies** - run `pip list | grep opencv`
4. **Check permissions** - camera access on macOS/Windows
5. **Monitor resources** - CPU/RAM usage during detection

The live camera system is designed to work out-of-the-box with your trained YOLOv12 model and provides a professional-grade fish detection interface for real-time monitoring and analysis.

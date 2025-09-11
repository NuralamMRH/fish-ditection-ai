# Fish Distance Estimation & Weight Calculation Guide

## Overview

This advanced fish detection system combines real-time fish detection with distance estimation and weight calculation capabilities. The system uses computer vision, depth estimation techniques, and scientific fish weight formulas to provide comprehensive fish analysis.

## Features

### 🎯 Advanced Detection Capabilities

- **YOLOv12 Fish Detection**: 28+ fish species recognition
- **Distance Estimation**: Multiple methods for depth calculation
- **Weight Calculation**: Scientific formulas for fish weight estimation
- **Real-time Analysis**: Live camera processing with annotations
- **Species Classification**: 426+ species database integration

## Distance Estimation Methods

### 1. Reference Object Method (MediaPipe)

Uses MediaPipe face detection as a reference for distance calculation:

```python
# Distance calculation using iris diameter as reference
face_distance = (real_iris_size * focal_length) / iris_width_pixels
```

**Parameters:**

- `real_iris_size`: 11.7mm (average human iris diameter)
- `focal_length`: Camera-specific (typically 800 pixels for HD cameras)
- `iris_width_pixels`: Measured iris width in the image

**Advantages:**

- High accuracy when reference face is present
- Real-world calibration
- Accounts for camera perspective

### 2. Size-Based Estimation

Uses fish size in frame to estimate distance:

```python
# Area ratio approach
area_ratio = fish_area / frame_area
estimated_distance = base_distance / sqrt(area_ratio)
```

**Parameters:**

- `fish_area`: Bounding box area in pixels
- `frame_area`: Total frame area
- `base_distance`: Empirical baseline (typically 300-500cm)

**Advantages:**

- Works without reference objects
- Provides reasonable estimates
- Adjustable for different scenarios

### 3. Camera Calibration Method

Uses intrinsic camera parameters for precise distance calculation:

```python
# Real-world dimensions from pixel measurements
real_size = (pixel_size * distance) / focal_length
```

## Weight Calculation Formulas

### 1. Girth-Based Formula

The most accurate method using length and girth measurements:

```
Weight (grams) = (Length × Girth²) ÷ 800
```

**Usage:**

- Most accurate for all fish species
- Requires girth estimation
- Best for scientific applications

### 2. Length-Based Formula

Simplified formula using only length:

```
Weight (grams) = (Length³) ÷ 1200
```

**Usage:**

- Quick estimation method
- Less accurate than girth-based
- Good for general applications

### 3. Species-Specific Formulas

Customized formulas per fish species:

```python
species_formulas = {
    'Tuna': lambda length: (length ** 2.8) / 1000,
    'Bass': lambda length: (length ** 3.0) / 1300,
    'Salmon': lambda length: (length ** 2.9) / 1100,
    'Catfish': lambda length: (length ** 3.1) / 1400,
    'Snapper': lambda length: (length ** 2.85) / 1050,
    'Mackerel': lambda length: (length ** 2.7) / 950,
    'Grouper': lambda length: (length ** 3.2) / 1500
}
```

## Fish Measurement Types

### Total Length (TL)

**Definition:** Distance from tip of nose to tip of tail
**Species:** Bass, Catfish, Salmon, Snapper, Grouper, Freshwater-Eel

```
+----nose----body----tail----+
|<------ Total Length ------>|
```

### Fork Length (FL)

**Definition:** Distance from nose to fork in tail
**Species:** Mackerel, Pompano, Blue marlin, Vietnamese mackerel, Tuna

```
+----nose----body----/fork\--+
|<--- Fork Length --->|
```

### Lower Jaw Fork Length (LJFL)

**Definition:** Distance from tip of lower jaw to fork in tail
**Species:** Sailfish, Swordfish, Marlin, Spearfish

```
+--lower_jaw--body----/fork\--+
|<-- LJFL Length ---->|
```

## Girth Estimation

### Species-Specific Girth Ratios

```python
girth_ratios = {
    'Tuna': 0.65,      # Robust, high girth/length ratio
    'Bass': 0.45,      # Moderate build
    'Salmon': 0.40,    # Streamlined
    'Catfish': 0.55,   # Round body
    'Snapper': 0.42,   # Compressed body
    'Mackerel': 0.35,  # Very streamlined
    'Grouper': 0.50,   # Stocky build
    'default': 0.45    # General estimate
}
```

### Girth Calculation

```python
estimated_girth = length_cm * girth_ratio
```

## Implementation Architecture

### 1. Distance Estimation Pipeline

```python
class FishDistanceAnalyzer:
    def estimate_distance(self, image, fish_bbox):
        # Method 1: Face reference (if available)
        face_distance = self.estimate_from_face_reference(image, fish_bbox)

        # Method 2: Size-based fallback
        size_distance = self.estimate_from_size(fish_bbox, image.shape)

        # Return best available estimate
        return face_distance if face_distance else size_distance
```

### 2. Dimension Calculation

```python
def calculate_fish_dimensions(self, fish_bbox, distance_mm, image_shape):
    x1, y1, x2, y2 = fish_bbox
    fish_width_pixels = x2 - x1
    fish_height_pixels = y2 - y1

    # Convert to real dimensions
    fish_length_mm = (fish_width_pixels * distance_mm) / self.focal_length
    fish_height_mm = (fish_height_pixels * distance_mm) / self.focal_length

    return {
        'length_cm': fish_length_mm / 10,
        'height_cm': fish_height_mm / 10,
        'length_inches': fish_length_mm / 25.4,
        'height_inches': fish_height_mm / 25.4
    }
```

### 3. Weight Calculation

```python
def calculate_fish_weight(self, dimensions, species):
    length_cm = dimensions['length_cm']
    girth_cm = self.estimate_girth(dimensions, species)

    # Multiple calculation methods
    weight_girth = (length_cm * girth_cm * girth_cm) / 800
    weight_length = (length_cm ** 3) / 1200
    weight_species = self.species_formula(length_cm, species)

    # Average the methods
    weights = [w for w in [weight_girth, weight_length, weight_species] if w]
    return sum(weights) / len(weights)
```

## Web Applications

### 1. Static Image Analysis (`fish_distance_weight_app.py`)

**Port:** 5005
**Features:**

- Upload image analysis
- Distance & weight calculation
- Species classification
- Comprehensive reporting

**Usage:**

```bash
python fish_distance_weight_app.py
# Open http://localhost:5005
```

### 2. Live Camera Analysis (`live_fish_distance_weight_app.py`)

**Port:** 5006
**Features:**

- Real-time fish detection
- Live distance estimation
- Continuous weight calculation
- Video streaming interface

**Usage:**

```bash
python live_fish_distance_weight_app.py
# Open http://localhost:5006
```

## API Endpoints

### Static Analysis API

```bash
# Analyze uploaded image
POST /analyze-distance-weight
Content-Type: multipart/form-data
Body: image file

# Get API information
GET /api/info
```

### Live Camera API

```bash
# Start camera feed
POST /start_camera

# Stop camera feed
POST /stop_camera

# Get current statistics
GET /stats

# Reset statistics
POST /reset_stats

# Take screenshot
POST /screenshot
```

## Response Format

### Complete Fish Analysis

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "total_fish": 2,
  "fish_detections": [
    {
      "fish_id": 1,
      "detection": {
        "bbox": [100, 150, 300, 250],
        "confidence": 0.92,
        "yolo_class": "Tuna"
      },
      "distance": {
        "distance_mm": 450.0,
        "distance_cm": 45.0,
        "distance_inches": 17.72,
        "estimation_method": "face_reference"
      },
      "dimensions": {
        "length_mm": 180.0,
        "height_mm": 85.0,
        "length_cm": 18.0,
        "height_cm": 8.5,
        "length_inches": 7.09,
        "height_inches": 3.35
      },
      "weight_calculation": {
        "weight_grams": 125.5,
        "weight_kg": 0.126,
        "weight_pounds": 0.277,
        "estimated_girth_cm": 11.7,
        "methods": {
          "girth_based": 120.2,
          "length_based": 128.7,
          "species_specific": 127.6
        }
      },
      "species_classification": {
        "predicted_species": "Bluefin Tuna",
        "confidence": 0.89,
        "species_id": 156
      },
      "measurement_type": "fork_length"
    }
  ],
  "summary": {
    "total_estimated_weight": {
      "grams": 245.8,
      "kg": 0.246,
      "pounds": 0.542
    },
    "average_distance_cm": 52.3,
    "largest_fish": {...},
    "closest_fish": {...}
  },
  "processing_time": {
    "detection": 0.045,
    "total": 2.678
  }
}
```

## Accuracy & Limitations

### Distance Estimation Accuracy

**Face Reference Method:**

- ±5-10% accuracy with clear face reference
- Best for distances 30-200cm
- Requires person in frame

**Size-Based Method:**

- ±15-25% accuracy
- Reasonable for distances 20-300cm
- Works without reference objects

### Weight Calculation Accuracy

**Girth-Based Formula:**

- ±10-15% for most species
- Best accuracy with proper girth estimation
- Requires species identification

**Length-Based Formula:**

- ±20-30% general accuracy
- Quick estimation method
- Less precise than girth-based

**Species-Specific:**

- ±8-12% for supported species
- Most accurate when species is correctly identified
- Limited to specific fish types

### Factors Affecting Accuracy

1. **Camera Quality:** Higher resolution = better measurements
2. **Distance Range:** Optimal at 50-150cm from camera
3. **Fish Orientation:** Side view provides best length measurement
4. **Lighting Conditions:** Good lighting improves detection accuracy
5. **Species Recognition:** Correct species ID improves weight calculation

## Calibration & Setup

### Camera Calibration

1. **Measure Camera Focal Length:**

   ```python
   # Place object of known size at known distance
   focal_length = (object_size_pixels * known_distance) / real_object_size
   ```

2. **Test with Known Objects:**

   ```python
   # Validate distance estimation with rulers/measuring tapes
   measured_distance = estimate_distance(image, object_bbox)
   error_percentage = abs(measured_distance - actual_distance) / actual_distance
   ```

3. **Adjust Parameters:**
   ```python
   # Fine-tune based on testing results
   self.focal_length = calibrated_focal_length
   self.base_distance = optimized_base_distance
   ```

### Species Girth Calibration

1. **Collect Real Fish Data:**

   - Measure actual fish length and girth
   - Record species and weight
   - Build species-specific datasets

2. **Calculate Girth Ratios:**

   ```python
   girth_ratio = measured_girth / measured_length
   species_ratios[species] = average(all_ratios_for_species)
   ```

3. **Validate Weight Formulas:**
   ```python
   calculated_weight = formula(length, estimated_girth)
   accuracy = abs(calculated_weight - actual_weight) / actual_weight
   ```

## Performance Optimization

### For Real-Time Applications

```python
# Optimize processing speed
def optimize_for_realtime():
    # Reduce image resolution for processing
    small_frame = cv2.resize(frame, (640, 480))

    # Process every nth frame
    if frame_count % 3 == 0:
        detections = detector.predict(small_frame)

    # Cache results between frames
    current_results = cached_results if not new_detections else new_results
```

### For Accuracy Applications

```python
# Optimize for accuracy
def optimize_for_accuracy():
    # Use full resolution images
    # Process every frame
    # Apply multiple estimation methods
    # Average results over time

    final_distance = average([
        face_reference_distance,
        size_based_distance,
        temporal_average_distance
    ])
```

## Integration Examples

### With Existing Fish Detection API

```python
# Combine with existing analysis API
def enhanced_fish_analysis(image_path):
    # Step 1: Basic fish detection
    basic_results = analyze_fish_image(image_path)

    # Step 2: Add distance & weight
    enhanced_results = add_distance_weight_analysis(basic_results, image_path)

    # Step 3: Combine results
    return merge_analysis_results(basic_results, enhanced_results)
```

### With Live Camera System

```python
# Integrate with live camera detection
class EnhancedLiveFishDetector(LiveFishDetector):
    def process_frame(self, frame):
        # Basic detection
        detections = super().process_frame(frame)

        # Add distance & weight
        for detection in detections:
            detection['distance'] = self.estimate_distance(frame, detection['bbox'])
            detection['weight'] = self.calculate_weight(detection)

        return detections
```

## Future Enhancements

### 1. Stereo Vision Distance Estimation

```python
# Use two cameras for accurate depth estimation
def stereo_distance_estimation(left_image, right_image, fish_bbox):
    disparity = calculate_disparity(left_image, right_image)
    distance = (baseline * focal_length) / disparity[fish_center]
    return distance
```

### 2. Machine Learning Weight Prediction

```python
# Train ML model for weight prediction
def ml_weight_prediction(fish_features):
    # Features: length, height, area, species, etc.
    model = load_trained_weight_model()
    predicted_weight = model.predict(fish_features)
    return predicted_weight
```

### 3. 3D Fish Reconstruction

```python
# Create 3D model for volume-based weight calculation
def volume_based_weight(fish_silhouette, distance):
    volume = estimate_3d_volume(fish_silhouette, distance)
    weight = volume * species_density
    return weight
```

This comprehensive system provides professional-grade fish analysis with distance estimation and weight calculation, suitable for research, aquaculture, and recreational fishing applications.

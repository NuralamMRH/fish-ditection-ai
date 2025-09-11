# Fish Analysis API Documentation

## Overview

The Fish Analysis API provides comprehensive fish detection and species classification using your custom-trained YOLOv12 model combined with a state-of-the-art classification system. The API processes uploaded images and returns detailed analysis including fish detection, species identification, confidence scores, size metrics, and annotated images.

## Base URL

```
http://localhost:5004
```

## Authentication

No authentication required for this local API.

## CORS Support

✅ **Cross-Origin Resource Sharing (CORS) is enabled** for all origins and methods.

The API includes the following CORS headers:

- `Access-Control-Allow-Origin: *` - Allows requests from any domain
- `Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization`

This means you can call the API from:

- Web applications running on different ports (e.g., `localhost:3000`, `localhost:5173`)
- Different domains
- Browser-based JavaScript applications
- React, Vue, Angular, or any frontend framework

**Example browser usage:**

```javascript
// This will work from any domain
const response = await fetch("http://localhost:5004/analyze", {
  method: "POST",
  body: formData,
});
```

## Endpoints

### 1. POST /analyze

**Main endpoint for fish image analysis**

#### Request

- **Method:** POST
- **Content-Type:** multipart/form-data
- **Parameters:**
  - `image` (required): Image file to analyze
  - **Supported formats:** PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
  - **Max file size:** 16MB

#### Response Format

```json
{
  "success": true,
  "fish_count": 3,
  "detections": [
    {
      "fish_id": 1,
      "detection": {
        "detected_class": "Crucian carp",
        "confidence": 0.9234,
        "bounding_box": [123, 45, 278, 156],
        "model": "YOLOv12-Local-Trained"
      },
      "classification": {
        "species": "Carassius carassius",
        "accuracy": 0.9876,
        "species_id": 42,
        "common_name": "Crucian Carp",
        "scientific_name": "Carassius carassius"
      },
      "polygon": [
        [123, 45],
        [278, 45],
        [278, 156],
        [123, 156]
      ],
      "metrics": {
        "dimensions": {
          "width_pixels": 155,
          "height_pixels": 111,
          "area_pixels": 17205
        },
        "relative_size": {
          "width_ratio": 0.1552,
          "height_ratio": 0.1387,
          "area_ratio": 0.0215
        },
        "position": {
          "center_x": 200,
          "center_y": 100,
          "relative_center_x": 0.2004,
          "relative_center_y": 0.125
        },
        "size_category": "Small"
      },
      "processing_time_ms": 45.2
    }
  ],
  "image_info": {
    "width": 1000,
    "height": 800,
    "channels": 3,
    "size_mb": 1.2
  },
  "processing_time": {
    "detection_ms": 30.5,
    "classification_ms": 120.8,
    "total_ms": 165.7
  },
  "models_used": {
    "detection": "YOLOv12-Local-Trained (28 fish species)",
    "classification": "EmbeddingClassifier-v7-1 (426 species)"
  },
  "annotated_image": {
    "filename": "analysis_20250625_123456_abc12345.jpg",
    "url": "/static/analysis_20250625_123456_abc12345.jpg"
  },
  "timestamp": "2025-06-25T12:34:56.789Z"
}
```

#### Error Response

```json
{
  "success": false,
  "error": "Error description",
  "timestamp": "2025-06-25T12:34:56.789Z"
}
```

#### Response Fields Explanation

**Detection Object:**

- `detected_class`: Fish type detected by YOLOv12 model
- `confidence`: Detection confidence score (0-1)
- `bounding_box`: [x1, y1, x2, y2] coordinates
- `model`: Model used for detection

**Classification Object:**

- `species`: Classified species name
- `accuracy`: Classification confidence (0-1)
- `species_id`: Internal species identifier
- `common_name`: Common name of the species
- `scientific_name`: Scientific name of the species

**Metrics Object:**

- `dimensions`: Pixel dimensions of detected fish
- `relative_size`: Size relative to image dimensions
- `position`: Fish position in image
- `size_category`: Small/Medium/Large/Very Small

### 2. GET /health

**API health check**

#### Response

```json
{
  "status": "healthy",
  "service": "Fish Analysis API",
  "models": {
    "yolo_v12": "loaded",
    "classifier": "available"
  },
  "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"],
  "max_file_size_mb": 16,
  "timestamp": "2025-06-25T12:34:56.789Z"
}
```

### 3. GET /models

**Model information**

#### Response

```json
{
  "detection_model": {
    "model_type": "YOLOv12",
    "training": "Local Custom Training",
    "species_count": 28,
    "species": [
      "Anchovies",
      "Bangus",
      "Basa fish",
      "Big-Head-Carp",
      "Black-Spotted-Barb",
      "Blue marlin",
      "Catfish"
      // ... full list of 28 species
    ],
    "confidence_threshold": 0.3
  },
  "classification_model": {
    "model_type": "EmbeddingClassifier",
    "version": "v7-1",
    "species_count": 426,
    "location": "./classification_rectangle_v7-1/"
  },
  "timestamp": "2025-06-25T12:34:56.789Z"
}
```

### 4. GET /static/{filename}

**Download annotated images**

Returns the annotated image file with fish detection overlays.

### 5. GET /

**API information and usage guide**

#### Response

```json
{
  "service": "Fish Analysis API",
  "version": "1.0.0",
  "description": "Comprehensive fish detection and analysis using YOLOv12 + Classification",
  "endpoints": {
    "POST /analyze": "Upload image for fish analysis",
    "GET /health": "API health check",
    "GET /models": "Model information",
    "GET /static/<filename>": "Download annotated images"
  },
  "usage": {
    "upload_method": "POST multipart/form-data",
    "field_name": "image",
    "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"],
    "max_size_mb": 16
  },
  "example_curl": "curl -X POST -F 'image=@fish.jpg' http://localhost:5004/analyze",
  "timestamp": "2025-06-25T12:34:56.789Z"
}
```

## Usage Examples

### cURL Examples

1. **Analyze an image:**

```bash
curl -X POST -F 'image=@fish_photo.jpg' http://localhost:5004/analyze
```

2. **Health check:**

```bash
curl http://localhost:5004/health
```

3. **Get model information:**

```bash
curl http://localhost:5004/models
```

4. **Download annotated image:**

```bash
curl -O http://localhost:5004/static/analysis_20250625_123456_abc12345.jpg
```

### Python Example

```python
import requests

# Analyze image
with open('fish_image.jpg', 'rb') as file:
    response = requests.post(
        'http://localhost:5004/analyze',
        files={'image': file}
    )

if response.status_code == 200:
    result = response.json()
    if result['success']:
        print(f"Found {result['fish_count']} fish!")
        for fish in result['detections']:
            detection = fish['detection']
            classification = fish['classification']
            print(f"Fish {fish['fish_id']}: {classification['species']} "
                  f"(YOLOv12: {detection['detected_class']}, "
                  f"Confidence: {detection['confidence']:.3f})")
    else:
        print(f"Error: {result['error']}")
else:
    print(f"HTTP Error: {response.status_code}")
```

### JavaScript Example

```javascript
// Analyze image using FormData
const analyzeImage = async (imageFile) => {
  const formData = new FormData();
  formData.append("image", imageFile);

  try {
    const response = await fetch("http://localhost:5004/analyze", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (result.success) {
      console.log(`Found ${result.fish_count} fish!`);
      result.detections.forEach((fish) => {
        console.log(`Fish ${fish.fish_id}: ${fish.classification.species}`);
      });
    } else {
      console.error("Analysis failed:", result.error);
    }
  } catch (error) {
    console.error("Request failed:", error);
  }
};
```

## Error Codes

| Status Code | Description                                    |
| ----------- | ---------------------------------------------- |
| 200         | Success                                        |
| 400         | Bad Request (invalid file, missing parameters) |
| 500         | Internal Server Error                          |

## Common Error Messages

- `"No image file provided. Use 'image' as the form field name."`
- `"Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff, webp"`
- `"YOLOv12 detector not available"`
- `"Could not load image"`
- `"Analysis failed: [specific error]"`

## Performance Notes

- **Average processing time:** 100-300ms per image
- **Detection speed:** ~30ms (YOLOv12)
- **Classification speed:** ~100-200ms (depends on fish count)
- **Memory usage:** ~2GB for models
- **Concurrent requests:** Supported (Flask threading)

## Model Information

### YOLOv12 Detection Model

- **Species detected:** 28 fish types
- **Training:** Custom local training on your dataset
- **Confidence threshold:** 0.3 (adjustable)
- **Input:** Any image size (auto-resized)

### Classification Model

- **Species classified:** 426 fish species
- **Model type:** EmbeddingClassifier v7-1
- **Accuracy:** High precision species identification
- **Input:** Fish crops from YOLO detection

## Rate Limiting

No rate limiting implemented for local use. For production deployment, consider implementing rate limiting based on your requirements.

## File Size Limits

- **Maximum file size:** 16MB
- **Recommended:** Under 5MB for optimal performance
- **Minimum resolution:** 64x64 pixels
- **Recommended resolution:** 640x640 or higher

## Integration Tips

1. **Batch Processing:** Send multiple requests for batch analysis
2. **Error Handling:** Always check the `success` field in responses
3. **Performance:** Cache results for identical images
4. **Storage:** Download and store annotated images if needed
5. **Monitoring:** Use `/health` endpoint for system monitoring

## Testing

Use the provided test client:

```bash
python test_analysis_api.py
```

This will run comprehensive tests including:

- Health checks
- Model information retrieval
- Image analysis
- Error handling
- Performance testing

## Support

For issues or questions about the API:

1. Check the `/health` endpoint for system status
2. Verify model files are properly loaded
3. Ensure image formats are supported
4. Review error messages for specific issues

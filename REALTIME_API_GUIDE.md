# 🐟 Real-time Fish Detection API with Callback System

## Overview

This API provides advanced fish detection using multiple YOLO models (YOLO12 + YOLO10) with 3D measurement, weight calculation, and callback URL integration. It's designed for real-time applications where fish data needs to be submitted to external systems.

## Features

- **Multi-Model Detection**: YOLO12 + YOLO10 fallback system for better detection rates
- **3D Measurement System**: Accurate length, weight, and dimension calculations
- **Species Classification**: Advanced fish species identification
- **Callback System**: Submit results to external URLs with automatic redirection
- **Query Parameter Support**: Easy callback URL integration via URL parameters
- **Session Management**: Track detection sessions and results
- **Real-time API**: Fast response times for live applications
- **REST API Submission**: Comprehensive data submission to callback URLs

## Quick Start

### 1. Start the API Server

```bash
python realtime_fish_api.py
```

The API will be available at:

- **API Base**: `http://localhost:5009`
- **Demo Interface**: `http://localhost:5009/demo`
- **Health Check**: `http://localhost:5009/api/health`

### 2. Test with Callback URL via Query Parameter

**Easy way to test with callback URL:**

```
http://localhost:5009/demo?callback_url=https://www.itrucksea.com/fishing-log/batch
```

This automatically sets the callback URL and enables direct submission to your system.

## API Endpoints

### Core Detection Endpoints

#### `POST /api/detect`

Upload image for fish detection and analysis.

**Method 1: Query Parameter (Recommended)**

```bash
curl -X POST \
  -F "image=@fish_photo.jpg" \
  "http://localhost:5009/api/detect?callback_url=https://www.itrucksea.com/fishing-log/batch"
```

**Method 2: Form Data**

```bash
curl -X POST \
  -F "image=@fish_photo.jpg" \
  -F "callback_url=https://www.itrucksea.com/fishing-log/batch" \
  http://localhost:5009/api/detect
```

**Method 3: Header**

```bash
curl -X POST \
  -F "image=@fish_photo.jpg" \
  -H "X-Callback-URL: https://www.itrucksea.com/fishing-log/batch" \
  http://localhost:5009/api/detect
```

**Response:**

```json
{
  "session_id": "uuid-here",
  "success": true,
  "fish_count": 2,
  "fish": [
    {
      "id": 0,
      "species": "Catfish",
      "confidence": 0.85,
      "bbox": [100, 150, 300, 400],
      "distance_cm": 85.2,
      "dimensions": {
        "total_length_cm": 45.3,
        "body_height_cm": 12.8,
        "estimated_girth_cm": 20.4,
        "length_inches": 17.8
      },
      "weight": {
        "weight_kg": 2.456,
        "weight_pounds": 5.41,
        "weight_grams": 2456.0
      },
      "detected_by": "YOLO12",
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "callback_url": "https://www.itrucksea.com/fishing-log/batch",
  "submit_url": "/api/submit/uuid-here",
  "models_used": ["YOLO12"]
}
```

#### `GET /api/results/{session_id}`

Get detection results for a specific session.

**Request:**

```bash
curl http://localhost:5009/api/results/your-session-id
```

### Callback Management

#### `POST /api/submit/{session_id}`

Submit results to callback URL via REST API.

**Browser Redirect (Default):**

```bash
# This will redirect to callback URL after submission
curl -X POST http://localhost:5009/api/submit/your-session-id
```

**API-only Response:**

```bash
# Add ?format=json to get JSON response instead of redirect
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Fisher Name", "location": "Lake Michigan"}' \
  "http://localhost:5009/api/submit/your-session-id?format=json"
```

**Response (API-only):**

```json
{
  "success": true,
  "session_id": "your-session-id",
  "callback_submitted": true,
  "callback_response": {
    "status_code": 200,
    "response_text": "Data received successfully",
    "headers": { "content-type": "application/json" }
  },
  "callback_error": null,
  "fish_count": 2,
  "redirect_url": "https://www.itrucksea.com/fishing-log/batch"
}
```

### Status & Health

#### `GET /api/status`

Get real-time API status.

#### `GET /api/health`

API health check with model information.

#### `GET /api/models`

Get detailed model information.

### Camera Endpoints (For Real-time Applications)

#### `POST /api/camera/start`

Start camera feed for real-time detection.

#### `POST /api/camera/stop`

Stop camera feed.

#### `GET /api/camera/feed`

Get camera feed status.

## Callback System Usage

### Basic Workflow with Query Parameters

1. **Upload Image with Callback URL via Query Parameter**

   ```bash
   curl -X POST \
     -F "image=@fish.jpg" \
     "http://localhost:5009/api/detect?callback_url=https://your-system.com/fish-data"
   ```

2. **Review Results**

   - API returns detection results with session ID
   - Fish data includes species, weight, dimensions, etc.

3. **Submit to Callback (Browser Redirect)**

   ```
   # User visits this URL and gets redirected to callback URL
   http://localhost:5009/api/submit/{session_id}
   ```

4. **Submit to Callback (API Only)**

   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"user_info": "John Doe", "location": "Lake Michigan"}' \
     "http://localhost:5009/api/submit/{session_id}?format=json"
   ```

### Enhanced Callback Data Format

When submitting to your callback URL, the API sends comprehensive data:

```json
{
  "session_id": "uuid-here",
  "timestamp": "2024-01-01T12:00:00",
  "fish_data": {
    "success": true,
    "fish_count": 2,
    "fish": [
      /* detailed fish array */
    ]
  },
  "source": "fish_detection_api",
  "version": "2.0",
  "api_endpoint": "http://localhost:5009/api/results/uuid-here",
  "total_fish_count": 2,
  "detection_success": true,
  "fish_summary": [
    {
      "species": "Catfish",
      "confidence": 0.85,
      "weight_kg": 2.456,
      "weight_pounds": 5.41,
      "length_cm": 45.3,
      "length_inches": 17.8,
      "detected_by": "YOLO12"
    }
  ],
  "total_weight_kg": 4.912,
  "total_weight_pounds": 10.82,
  "user_data": {
    "user_info": "Additional user data if provided"
  }
}
```

## Integration Examples

### Web Application Integration

**HTML with Query Parameter:**

```html
<!-- Auto-load callback URL from query parameter -->
<script>
  const urlParams = new URLSearchParams(window.location.search);
  const callbackUrl = urlParams.get("callback_url");

  if (callbackUrl) {
    // Use callback URL in API calls
    fetch(`/api/detect?callback_url=${encodeURIComponent(callbackUrl)}`, {
      method: "POST",
      body: formData,
    });
  }
</script>
```

**Direct Redirect Method:**

```javascript
// For browser redirect after submission
function submitAndRedirect(sessionId) {
  // This will redirect to callback URL
  window.location.href = `/api/submit/${sessionId}`;
}

// For API-only submission
function submitToAPI(sessionId) {
  fetch(`/api/submit/${sessionId}?format=json`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_info: "Web User" }),
  })
    .then((response) => response.json())
    .then((data) => console.log("Callback result:", data));
}
```

### Python Integration

```python
import requests

# Method 1: Query parameter (recommended)
files = {'image': open('fish.jpg', 'rb')}
callback_url = 'https://your-system.com/fish-data'

response = requests.post(
    f'http://localhost:5009/api/detect?callback_url={callback_url}',
    files=files
)
result = response.json()

if result['success']:
    session_id = result['session_id']
    print(f"Detected {result['fish_count']} fish")

    # Submit to callback (API mode)
    submit_data = {
        'user_info': 'Python Client',
        'fishing_trip_id': '12345'
    }

    submit_response = requests.post(
        f'http://localhost:5009/api/submit/{session_id}?format=json',
        json=submit_data
    )

    callback_result = submit_response.json()
    if callback_result['callback_submitted']:
        print("Data successfully submitted to callback URL!")
    else:
        print(f"Callback failed: {callback_result['callback_error']}")
```

### URL Examples

```bash
# Demo with callback URL
http://localhost:5009/demo?callback_url=https://www.itrucksea.com/fishing-log/batch

# API detection with callback
http://localhost:5009/api/detect?callback_url=https://your-system.com/api/fish-data

# Health check
http://localhost:5009/api/health

# Submit and redirect
http://localhost:5009/api/submit/SESSION_ID_HERE

# Submit API-only
http://localhost:5009/api/submit/SESSION_ID_HERE?format=json
```

## cURL Examples

```bash
# Health check
curl http://localhost:5009/api/health

# Detect fish with callback (query parameter method)
curl -X POST \
  -F "image=@fish.jpg" \
  "http://localhost:5009/api/detect?callback_url=https://www.itrucksea.com/fishing-log/batch"

# Submit results (API mode)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"user_info": "Test User", "location": "Test Lake"}' \
  "http://localhost:5009/api/submit/SESSION_ID_HERE?format=json"

# Submit results (redirect mode)
curl -X POST \
  -d '{}' \
  http://localhost:5009/api/submit/SESSION_ID_HERE
```

## Configuration

### Environment Variables

```bash
# API Configuration
export FISH_API_PORT=5009
export FISH_API_HOST=0.0.0.0

# Model Paths
export YOLO12_MODEL_PATH=./detector_v12
export YOLO10_MODEL_PATH=./detector_v10_m3
export CLASSIFIER_PATH=./classification_rectangle_v7-1

# Upload Limits
export MAX_FILE_SIZE=32MB
export SESSION_TIMEOUT=24h
```

### Custom Callback URLs

The API supports various callback URL patterns:

- **REST APIs**: `https://api.yoursite.com/fish-data`
- **Webhooks**: `https://hooks.yoursite.com/webhook/fish`
- **Form Submissions**: `https://yoursite.com/forms/fish-log`

## Error Handling

### Common Error Responses

```json
{
  "success": false,
  "error": "No image file provided",
  "code": "MISSING_IMAGE"
}
```

```json
{
  "success": false,
  "error": "No fish detected by any model",
  "fish_count": 0
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (missing data, invalid image)
- `404`: Session not found
- `500`: Server error

## Performance

### Typical Response Times

- **Fish Detection**: 200-500ms per image
- **Multi-model Fallback**: <100ms additional
- **Species Classification**: 50-100ms additional
- **3D Measurements**: <10ms

### Throughput

- **Concurrent Sessions**: 10-20 simultaneous
- **Images per minute**: 100-200 (depending on hardware)
- **Session Storage**: 1000+ active sessions

## Testing

### Run Test Suite

```bash
python test_realtime_api.py
```

### Test Modes

1. **Complete API Test**: Full workflow testing
2. **Callback Workflow Demo**: Specific callback testing
3. **Health Check Only**: Basic connectivity test

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5009

CMD ["python", "realtime_fish_api.py"]
```

### Production Considerations

- Use HTTPS for callback URLs
- Implement rate limiting
- Add authentication for sensitive endpoints
- Monitor session storage usage
- Set up logging and monitoring

## Troubleshooting

### Common Issues

1. **Models not loading**

   - Check model paths in directories
   - Verify CUDA/CPU compatibility

2. **Callback failures**

   - Verify callback URL is accessible
   - Check CORS settings if cross-domain

3. **Large file uploads**
   - Increase MAX_CONTENT_LENGTH
   - Check available disk space

### Debug Mode

```bash
export FLASK_DEBUG=1
python realtime_fish_api.py
```

## Support

- **Demo Interface**: `http://localhost:5009/demo`
- **API Documentation**: `http://localhost:5009/`
- **Health Status**: `http://localhost:5009/api/health`

## Advanced Features

### Batch Processing

Process multiple images in one request:

```python
files = [
    ('images', open('fish1.jpg', 'rb')),
    ('images', open('fish2.jpg', 'rb'))
]
response = requests.post('/api/detect_batch', files=files)
```

### Real-time Streaming

Connect to camera feed for live detection:

```python
import cv2
import requests

# Start camera
requests.post('/api/camera/start')

# Process frames
while True:
    response = requests.get('/api/camera/feed')
    if response.json()['detection_count'] > 0:
        print("Fish detected!")
```

### Custom Measurements

Override default measurement parameters:

```json
{
  "image": "base64_image_data",
  "measurement_config": {
    "pixels_per_cm": 50,
    "depth_estimation": "monocular",
    "species_weights": { "Catfish": 0.0003 }
  }
}
```

This API provides a complete solution for fish detection with seamless integration into existing fishing log systems!

# 🐟 Fish Identification API - Node.js Integration

This directory contains Node.js clients and examples for integrating with your fish identification API.

## 🎯 What You Have

- ✅ **Working Python API** running on `http://localhost:5001`
- ✅ **YOLOv10 Fish Detection** (97.9% confidence)
- ✅ **Species Classification** (426+ fish species with 100% accuracy)
- ✅ **REST API endpoints** for programmatic access

## 📁 Files in this Directory

| File                 | Description                         |
| -------------------- | ----------------------------------- |
| `fish_api_client.js` | Node.js client library for the API  |
| `express_server.js`  | Complete Express.js web application |
| `package.json`       | NPM dependencies and scripts        |
| `README.md`          | This documentation                  |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd nodejs_client
npm install
```

### 2. Make Sure Python API is Running

```bash
# In the main project directory
source venv/bin/activate
python run_web_app_fixed.py
```

The Python API should be running on `http://localhost:5001`

### 3. Test the Node.js Client

```bash
# Run the demo
npm run demo
```

### 4. Start the Express.js Web Server

```bash
# Install additional dependencies for the web server
npm install express multer

# Start the web server
node express_server.js
```

Open your browser: `http://localhost:3000`

## 💻 Usage Examples

### Basic API Client Usage

```javascript
const FishIdentificationClient = require("./fish_api_client");

const client = new FishIdentificationClient("http://localhost:5001");

// Check if API is ready
const health = await client.checkHealth();
console.log("API Status:", health.status);

// Identify fish in an image
const result = await client.identifyFish("../test_image.png");
if (result.success) {
  console.log(`Found ${result.fish_count} fish:`);
  result.fish.forEach((fish) => {
    console.log(`- ${fish.species} (${fish.accuracy} accuracy)`);
  });
}
```

### Express.js Integration

```javascript
const express = require("express");
const FishIdentificationClient = require("./fish_api_client");

const app = express();
const fishAPI = new FishIdentificationClient();

app.post("/identify", async (req, res) => {
  const result = await fishAPI.identifyFish(req.file.path);
  res.json(result);
});
```

### React Component Example

```jsx
import React, { useState } from "react";

function FishIdentifier() {
  const [result, setResult] = useState(null);

  const handleFileUpload = async (event) => {
    const formData = new FormData();
    formData.append("file", event.target.files[0]);

    const response = await fetch("http://localhost:5001/api", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <input type="file" onChange={handleFileUpload} />
      {result && result.success && (
        <div>
          <h3>Results:</h3>
          {result.fish.map((fish) => (
            <p key={fish.fish_id}>
              {fish.species} - {fish.accuracy} accuracy
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
```

## 🔗 API Endpoints

### Python API (localhost:5001)

| Endpoint  | Method | Description                          |
| --------- | ------ | ------------------------------------ |
| `/health` | GET    | Check API status and model health    |
| `/api`    | POST   | Upload image for fish identification |
| `/`       | GET    | Web interface for testing            |

### Node.js Web Server (localhost:3000)

| Endpoint        | Method | Description                      |
| --------------- | ------ | -------------------------------- |
| `/`             | GET    | Web interface with drag & drop   |
| `/api/health`   | GET    | Proxy to Python API health check |
| `/api/identify` | POST   | Proxy to fish identification     |

## 📊 API Response Format

### Success Response

```json
{
  "success": true,
  "fish_count": 1,
  "fish": [
    {
      "fish_id": 1,
      "species": "Tinca tinca",
      "accuracy": 1.0,
      "confidence": 0.979,
      "box": [0, 0, 609, 278]
    }
  ]
}
```

### Error Response

```json
{
  "error": "No fish detected in image"
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "models_ready": true,
  "yolo_detector": "✅ Working",
  "fish_classifier": "✅ Working",
  "api_version": "1.0"
}
```

## 🔧 Configuration Options

### Client Configuration

```javascript
const client = new FishIdentificationClient({
  apiUrl: "http://localhost:5001", // Python API URL
  timeout: 30000, // Request timeout (ms)
  retries: 3, // Number of retries
});
```

### Upload Limits

- **Max file size**: 16MB
- **Allowed formats**: PNG, JPG, JPEG, GIF, BMP
- **Processing time**: 3-10 seconds depending on image size

## 🚀 Deployment Options

### 1. Local Development

```bash
# Terminal 1: Python API
python run_web_app_fixed.py

# Terminal 2: Node.js Server
node express_server.js
```

### 2. Production Deployment

```bash
# Use PM2 for process management
npm install -g pm2

# Start Python API
pm2 start "python run_web_app_fixed.py" --name fish-api

# Start Node.js server
pm2 start express_server.js --name fish-web
```

### 3. Docker Deployment

```dockerfile
# Dockerfile example
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "express_server.js"]
```

## 🔍 Troubleshooting

### Common Issues

1. **"Cannot connect to Fish Identification API"**

   - Make sure Python API is running on localhost:5001
   - Check if virtual environment is activated
   - Verify all Python dependencies are installed

2. **"Models not ready"**

   - Check Python console for model loading errors
   - Ensure model files exist in detector_v10_m3/ and classification_rectangle_v7-1/
   - Wait for models to fully load (can take 30-60 seconds)

3. **"File too large"**

   - Resize images to under 16MB
   - Use image compression tools

4. **"No fish detected"**
   - Ensure image contains visible fish
   - Try different angles or lighting
   - Check image quality and resolution

### Debug Mode

```javascript
// Enable debug logging
const client = new FishIdentificationClient();
client.debug = true;

// Check detailed health status
const health = await client.checkHealth();
console.log("Detailed health:", health);
```

## 🎯 Next Steps

1. **Integrate into your application** using the client library
2. **Customize the UI** by modifying express_server.js
3. **Add authentication** for production use
4. **Implement caching** for better performance
5. **Add batch processing** for multiple images
6. **Set up monitoring** and logging

## 🔧 Development

### Adding New Features

```javascript
// Extend the client class
class CustomFishClient extends FishIdentificationClient {
  async identifyWithMetadata(imagePath, metadata) {
    const result = await this.identifyFish(imagePath);
    return {
      ...result,
      metadata,
      timestamp: new Date().toISOString(),
    };
  }
}
```

### Testing

```bash
# Run tests
npm test

# Test specific image
node -e "
const client = require('./fish_api_client');
const api = new client();
api.identifyFish('../test_image.png').then(console.log);
"
```

## 📞 Support

- **Python API Issues**: Check GETTING_STARTED.md in main directory
- **Node.js Integration**: See examples in this directory
- **Deployment**: Check DEPLOYMENT_GUIDE.md for production setup

**🎉 You now have a complete fish identification system with both Python and Node.js interfaces!**

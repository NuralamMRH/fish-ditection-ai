#!/usr/bin/env node
/**
 * Fish Identification API Client for Node.js
 * ==========================================
 *
 * This demonstrates how to use your fish identification API
 * from any Node.js application or web server.
 */

const fs = require("fs");
const FormData = require("form-data");
const fetch = require("node-fetch");
const path = require("path");

class FishIdentificationClient {
  constructor(apiUrl = "http://localhost:5001") {
    this.apiUrl = apiUrl;
  }

  /**
   * Check if the API is healthy and models are loaded
   */
  async checkHealth() {
    try {
      const response = await fetch(`${this.apiUrl}/health`);
      const health = await response.json();
      return health;
    } catch (error) {
      console.error("Health check failed:", error.message);
      return null;
    }
  }

  /**
   * Identify fish in an image
   * @param {string} imagePath - Path to the image file
   * @returns {Object} Fish identification results
   */
  async identifyFish(imagePath) {
    try {
      // Check if file exists
      if (!fs.existsSync(imagePath)) {
        throw new Error(`Image file not found: ${imagePath}`);
      }

      // Create form data
      const form = new FormData();
      form.append("file", fs.createReadStream(imagePath));

      // Send request
      const response = await fetch(`${this.apiUrl}/api`, {
        method: "POST",
        body: form,
      });

      const result = await response.json();
      return result;
    } catch (error) {
      console.error("Fish identification failed:", error.message);
      return { error: error.message };
    }
  }

  /**
   * Process multiple images in batch
   * @param {Array} imagePaths - Array of image file paths
   * @returns {Array} Array of identification results
   */
  async identifyBatch(imagePaths) {
    const results = [];

    console.log(`🐟 Processing ${imagePaths.length} images...`);

    for (let i = 0; i < imagePaths.length; i++) {
      const imagePath = imagePaths[i];
      console.log(
        `📷 Processing image ${i + 1}/${imagePaths.length}: ${path.basename(
          imagePath
        )}`
      );

      const result = await this.identifyFish(imagePath);
      results.push({
        image: imagePath,
        result: result,
      });

      // Small delay to avoid overwhelming the API
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    return results;
  }
}

/**
 * Demo function showing how to use the client
 */
async function demo() {
  console.log("🐟 Fish Identification API Client Demo");
  console.log("=" * 50);

  const client = new FishIdentificationClient();

  // Check API health
  console.log("🩺 Checking API health...");
  const health = await client.checkHealth();

  if (!health) {
    console.log("❌ API is not responding");
    console.log("💡 Make sure your Python server is running:");
    console.log("   python run_web_app_fixed.py");
    return;
  }

  console.log("✅ API Status:", health.status);
  console.log("📍 YOLO Detector:", health.yolo_detector);
  console.log("🔬 Fish Classifier:", health.fish_classifier);

  if (!health.models_ready) {
    console.log("❌ Models not ready");
    return;
  }

  // Test with a single image
  console.log("\n🔍 Testing fish identification...");
  const testImage = "../test_image.png";

  if (fs.existsSync(testImage)) {
    const result = await client.identifyFish(testImage);

    if (result.success) {
      console.log(`✅ Found ${result.fish_count} fish:`);
      result.fish.forEach((fish, index) => {
        console.log(`🐠 Fish ${fish.fish_id}:`);
        console.log(`   Species: ${fish.species}`);
        console.log(`   Accuracy: ${fish.accuracy}`);
        console.log(`   Confidence: ${fish.confidence}`);
        console.log(`   Location: [${fish.box.join(", ")}]`);
      });
    } else {
      console.log("❌ Error:", result.error);
    }
  } else {
    console.log("❌ Test image not found:", testImage);
  }
}

/**
 * Example usage in a web server (Express.js)
 */
function webServerExample() {
  return `
// Example: Using the Fish API client in an Express.js web server

const express = require('express');
const multer = require('multer');
const FishIdentificationClient = require('./fish_api_client');

const app = express();
const upload = multer({ dest: 'uploads/' });
const fishAPI = new FishIdentificationClient('http://localhost:5001');

// Endpoint to identify fish in uploaded images
app.post('/identify-fish', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }
        
        // Use your fish identification API
        const result = await fishAPI.identifyFish(req.file.path);
        
        // Return results to frontend
        res.json(result);
        
        // Clean up uploaded file
        fs.unlinkSync(req.file.path);
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => {
    console.log('Web server running on http://localhost:3000');
});
`;
}

/**
 * Example usage for React/Next.js frontend
 */
function frontendExample() {
  return `
// Example: Using the Fish API from a React component

import React, { useState } from 'react';

function FishIdentifier() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const handleImageUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('http://localhost:5001/api', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
        }
        
        setLoading(false);
    };
    
    return (
        <div>
            <h2>🐟 Fish Identification</h2>
            <input type="file" onChange={handleImageUpload} accept="image/*" />
            
            {loading && <p>Identifying fish...</p>}
            
            {result && result.success && (
                <div>
                    <h3>Results:</h3>
                    {result.fish.map(fish => (
                        <div key={fish.fish_id}>
                            <p><strong>Species:</strong> {fish.species}</p>
                            <p><strong>Accuracy:</strong> {fish.accuracy}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
`;
}

// Export the client class
module.exports = FishIdentificationClient;

// Run demo if this file is executed directly
if (require.main === module) {
  demo().catch(console.error);
}

// Show usage examples
console.log("\n💡 Usage Examples:");
console.log("\n🌐 Express.js Web Server:");
console.log(webServerExample());
console.log("\n⚛️ React Frontend:");
console.log(frontendExample());

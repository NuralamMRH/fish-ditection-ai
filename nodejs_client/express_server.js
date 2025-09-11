#!/usr/bin/env node
/**
 * Express.js Web Server with Fish Identification
 * ==============================================
 * 
 * A complete web application that uses your fish identification API
 */

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const FishIdentificationClient = require('./fish_api_client');

const app = express();
const port = 3000;

// Configure multer for file uploads
const upload = multer({
    dest: 'uploads/',
    limits: {
        fileSize: 16 * 1024 * 1024, // 16MB limit
    },
    fileFilter: (req, file, cb) => {
        // Allow only image files
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files allowed!'), false);
        }
    }
});

// Initialize fish API client
const fishAPI = new FishIdentificationClient('http://localhost:5001');

// Serve static files
app.use(express.static('public'));
app.use(express.json());

// HTML template for the web interface
const htmlTemplate = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐟 Fish Identification Service</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f8ff; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-form { margin: 20px 0; padding: 20px; border: 2px dashed #ddd; border-radius: 8px; text-align: center; }
        .upload-form.dragover { border-color: #2196F3; background: #f0f8ff; }
        .result { margin: 20px 0; padding: 15px; border-radius: 8px; }
        .success { background: #e8f5e8; border: 1px solid #4caf50; }
        .error { background: #ffebee; border: 1px solid #f44336; color: #c62828; }
        .fish-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #fafafa; }
        button { background: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976D2; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .loading { display: none; margin: 20px 0; text-align: center; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.healthy { background: #e8f5e8; }
        .status.error { background: #ffebee; }
        #preview { max-width: 300px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 Fish Identification Service</h1>
        <p>Upload fish images to identify species using advanced AI models</p>
        
        <div id="status" class="status">Checking API status...</div>
        
        <div class="upload-form" id="uploadForm">
            <p>📸 Drop image here or click to select</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Choose Image</button>
            <img id="preview" style="display: none;">
        </div>
        
        <div class="loading" id="loading">
            <p>🔍 Analyzing fish... Please wait...</p>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const status = document.getElementById('status');

        // Check API status on page load
        checkAPIStatus();

        async function checkAPIStatus() {
            try {
                const response = await fetch('/api/health');
                const health = await response.json();
                
                if (health.models_ready) {
                    status.className = 'status healthy';
                    status.innerHTML = '✅ API Ready - All models loaded successfully';
                } else {
                    status.className = 'status error';
                    status.innerHTML = '❌ API Error - Models not ready';
                }
            } catch (error) {
                status.className = 'status error';
                status.innerHTML = '❌ Cannot connect to Fish Identification API';
            }
        }

        // Handle drag and drop
        uploadForm.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadForm.classList.add('dragover');
        });

        uploadForm.addEventListener('dragleave', () => {
            uploadForm.classList.remove('dragover');
        });

        uploadForm.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadForm.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Upload and identify
            identifyFish(file);
        }

        async function identifyFish(file) {
            loading.style.display = 'block';
            results.innerHTML = '';

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/api/identify', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                displayResults(result);

            } catch (error) {
                displayResults({ error: 'Failed to identify fish: ' + error.message });
            }

            loading.style.display = 'none';
        }

        function displayResults(result) {
            if (result.error) {
                results.innerHTML = \`
                    <div class="result error">
                        <h3>❌ Error</h3>
                        <p>\${result.error}</p>
                    </div>
                \`;
                return;
            }

            if (result.success && result.fish_count > 0) {
                let html = \`
                    <div class="result success">
                        <h3>📊 Identification Results</h3>
                        <p><strong>Fish detected:</strong> \${result.fish_count}</p>
                \`;

                result.fish.forEach(fish => {
                    html += \`
                        <div class="fish-item">
                            <h4>🐠 Fish #\${fish.fish_id}</h4>
                            <p><strong>Species:</strong> \${fish.species}</p>
                            <p><strong>Classification Accuracy:</strong> \${fish.accuracy}</p>
                            <p><strong>Detection Confidence:</strong> \${fish.confidence}</p>
                            <p><strong>Location:</strong> [\${fish.box.join(', ')}]</p>
                        </div>
                    \`;
                });

                html += '</div>';
                results.innerHTML = html;
            } else {
                results.innerHTML = \`
                    <div class="result error">
                        <h3>🔍 No Fish Detected</h3>
                        <p>No fish were found in the uploaded image.</p>
                    </div>
                \`;
            }
        }
    </script>
</body>
</html>
`;

// Routes
app.get('/', (req, res) => {
    res.send(htmlTemplate);
});

// API route to check health
app.get('/api/health', async (req, res) => {
    try {
        const health = await fishAPI.checkHealth();
        res.json(health || { status: 'error', models_ready: false });
    } catch (error) {
        res.status(500).json({ status: 'error', error: error.message });
    }
});

// API route to identify fish
app.post('/api/identify', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }

        // Check file size
        const stats = fs.statSync(req.file.path);
        if (stats.size > 16 * 1024 * 1024) {
            fs.unlinkSync(req.file.path);
            return res.status(400).json({ error: 'File too large (max 16MB)' });
        }

        // Use fish identification API
        const result = await fishAPI.identifyFish(req.file.path);

        // Clean up uploaded file
        fs.unlinkSync(req.file.path);

        res.json(result);

    } catch (error) {
        console.error('Identification error:', error);
        
        // Clean up file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        res.status(500).json({ error: error.message });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large (max 16MB)' });
        }
    }
    
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(port, async () => {
    console.log(\`🌐 Fish Identification Web Server started!\`);
    console.log(\`📱 Open your browser: http://localhost:\${port}\`);
    console.log(\`🔗 API endpoint: http://localhost:\${port}/api/identify\`);
    
    // Check if Python API is running
    console.log(\`\\n🔍 Checking connection to Python API...\`);
    const health = await fishAPI.checkHealth();
    
    if (health && health.models_ready) {
        console.log(\`✅ Connected to Fish Identification API\`);
        console.log(\`📍 YOLO Detector: \${health.yolo_detector}\`);
        console.log(\`🔬 Fish Classifier: \${health.fish_classifier}\`);
    } else {
        console.log(\`❌ Cannot connect to Python API at http://localhost:5001\`);
        console.log(\`💡 Make sure to run: python run_web_app_fixed.py\`);
    }
    
    console.log(\`\\n🎉 Ready to identify fish!\`);
}); 
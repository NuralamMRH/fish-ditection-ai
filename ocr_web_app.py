#!/usr/bin/env python3
"""
Enhanced OCR Web App: Document Text Detection with Structured Field Extraction
===========================================================================
- Upload document images and extract raw text using pytesseract
- Extract structured key-value pairs from Vietnamese vessel registration documents
- Modern HTML interface with both raw text and structured data display
"""

import os
import re
from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract  # type: ignore

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def ocr_image_lines(image_path):
    try:
        # Open and preprocess the image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Configure tesseract parameters for better Vietnamese text recognition
        custom_config = r'--oem 3 --psm 3 -l vie+eng'
        
        # Get both raw text and lines
        text = pytesseract.image_to_string(img, config=custom_config)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        print(f"[OCR] Successfully processed image: {image_path}")
        print(f"[OCR] Found {len(lines)} text lines")
        print(f"[OCR] Raw text: {text}")
        
        return lines, text
    except Exception as e:
        print(f"[OCR] Error processing image {image_path}: {str(e)}")
        return [], ""

def extract_structured_fields(text):
    """
    Extract structured key-value pairs from Vietnamese vessel registration documents
    """
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Print the full text for debugging
    print(f"[DEBUG] Full OCR text: {text}")
    
    # Initialize result dictionary with additional fishing vessel specific fields
    result = {
        'certificate_number': '',
        'vessel_type': '',
        'gross_tonnage': '',
        'dead_weight': '',
        'length_overall': '',
        'length_design': '',
        'breadth_overall': '',
        'breadth_design': '',
        'depth': '',
        'draught': '',
        'hull_material': '',
        'construction_place': '',
        'engine_count': '',
        'engine_type': '',
        'engine_number': '',
        'total_power': '',
        'power': '',
        'engine_manufacture_place': '',
        'port_registry': '',
        'registration_number': '',
        'registration_type': '',
        'issue_date': '',
        'issue_place': '',
        'valid_until': ''
    }
    
    # More specific patterns for fishing vessel certificates
    patterns = {
        'certificate_number': [
            r'(?:Số|No)[:\s]*([0-9]+)',
            r'(?:Number)[:\s]*([0-9]+)'
        ],
        'vessel_type': [
            r'(?:Kiểu\s*tàu|Type\s*of\s*Vessel)[:\s]*([A-Z0-9\-]+\/[A-Z]+)',
        ],
        'gross_tonnage': [
            r'(?:Tổng\s*dung\s*tích|Gross\s*Tonnage)[,\s]*GT[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'dead_weight': [
            r'(?:Trọng\s*tải\s*toàn\s*phần|Dead\s*weight)[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'length_overall': [
            r'(?:Chiều\s*dài\s*Lmax|Length\s*overall)[,\s]*m[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'length_design': [
            r'(?:Chiều\s*dài\s*thiết\s*kế\s*Lk|Length)[,\s]*m[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'breadth_overall': [
            r'(?:Chiều\s*rộng\s*Bmax|Breadth\s*overall)[,\s]*m[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'breadth_design': [
            r'(?:Chiều\s*rộng\s*thiết\s*kế\s*Bk|Breadth)[,\s]*m[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'depth': [
            r'(?:Chiều\s*chìm\s*d|Depth)[,\s]*m[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'draught': [
            r'(?:Chiều\s*cao\s*mạn\s*D|Draught)[,\s]*m[:\s]*([0-9]+[,\.]?[0-9]*)',
        ],
        'hull_material': [
            r'(?:Vật\s*liệu\s*vỏ|Materials)[:\s]*([A-Za-zÀ-ỹ]+)',
        ],
        'construction_place': [
            r'(?:Nơi\s*và\s*nơi\s*đóng|Year\s*and\s*Place\s*of\s*Build)[:\s]*([A-Za-zÀ-ỹ\s]+)',
        ],
        'engine_count': [
            r'(?:Số\s*lượng\s*máy|Number\s*of\s*Engines)[:\s]*([0-9]+)',
        ],
        'engine_type': [
            r'(?:Ký\s*hiệu\s*máy|Type\s*of\s*machine)[:\s]*([A-Za-z0-9\s]+)',
        ],
        'engine_number': [
            r'(?:Số\s*máy|Number\s*engines)[:\s]*([0-9]+)',
        ],
        'total_power': [
            r'(?:Tổng\s*công\s*suất|Total\s*power)\s*\(KW\)[:\s]*([0-9]+)',
        ],
        'power': [
            r'(?:Công\s*suất|Power)\s*\(KW\)[:\s]*([0-9]+)',
        ],
        'engine_manufacture_place': [
            r'(?:Nơi\s*và\s*năm\s*chế\s*tạo|Year\s*and\s*place\s*of\s*manufacture)[:\s]*([A-Za-zÀ-ỹ\s]+)',
        ],
        'port_registry': [
            r'(?:Cảng\s*đăng\s*ký|Port\s*Registry)[,\s]*(?:Hô\s*Gọi)?[:\s]*([A-Za-zÀ-ỹ\s]+)',
        ],
        'registration_number': [
            r'(?:Số\s*đăng\s*ký|Number\s*or\s*registry)[:\s]*(CM-\s*[0-9]+\s*-TS)',
        ],
        'registration_type': [
            r'(?:Loại\s*đăng\s*ký|Type\s*of\s*registration)[:\s]*([A-Z]+)',
        ],
        'issue_date': [
            r'(?:Cấp\s*tại\s*Cà\s*Mau|Issued\s*at\s*Ca\s*Mau)[,\s]*(?:ngày|Date)[:\s]*([0-9]{1,2}[\/\s]+[0-9]{1,2}[\/\s]+[0-9]{4})',
        ],
        'valid_until': [
            r'(?:Giấy\s*chứng\s*nhận\s*này\s*có\s*hiệu\s*lực\s*đến|This\s*certificate\s*is\s*valid\s*until)[:\s]*([A-Za-zÀ-ỹ\s]+)',
        ]
    }
    
    # Extract fields using regex patterns
    for field, pattern_list in patterns.items():
        for i, pattern in enumerate(pattern_list):
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and match.group(1).strip():
                    extracted_value = match.group(1).strip()
                    print(f"[DEBUG] Field '{field}' matched pattern {i+1}: '{extracted_value}'")
                    result[field] = extracted_value
                    break
            except Exception as e:
                print(f"[DEBUG] Error in pattern {i+1} for field '{field}': {e}")
                continue
    
    # Clean up extracted values
    for key, value in result.items():
        if value:
            # Remove extra whitespace
            result[key] = re.sub(r'\s+', ' ', value).strip()
            # Remove common OCR artifacts
            result[key] = re.sub(r'[|]', '', result[key])
            # Remove leading/trailing punctuation
            result[key] = re.sub(r'^[:\-\s]+|[:\-\s]+$', '', result[key])
    
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template_string(TEMPLATE)

@app.route('/test', methods=['GET'])
def test_extraction():
    """Test route to debug field extraction with sample text"""
    sample_text = """
    Tên: Nguyễn Văn An
    Số CMND: 123456789
    Địa chỉ: 123 Đường ABC, Quận 1, TP.HCM
    Tên tàu: Tàu Cá Số 1
    Loại tàu: Tàu cá
    Số đăng ký: TC-12345
    Ngày đăng ký: 15/06/2023
    Chiều dài: 12.5 m
    Chiều rộng: 3.2 m
    Tổng dung tích: 15.8
    Máy chính: 120 HP
    Cảng đăng ký: Cảng Vũng Tàu
    """
    
    print(f"[TEST] Testing extraction with sample text")
    structured_data = extract_structured_fields(sample_text)
    extracted_fields = {k: v for k, v in structured_data.items() if v}
    
    return jsonify({
        'sample_text': sample_text,
        'structured_data': structured_data,
        'extracted_fields': extracted_fields,
        'extracted_count': len(extracted_fields)
    })

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if not file or not file.filename or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format. Please upload PNG, JPG, JPEG, BMP, TIFF, or WEBP.'}), 400
        
        # Save the file with proper error handling
        try:
            filename = secure_filename(file.filename if file.filename else 'uploaded_image.png')
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Verify the file was saved
            if not os.path.exists(filepath):
                return jsonify({'success': False, 'error': 'Failed to save uploaded file'}), 500
                
            print(f"[OCR] Saved image to: {filepath}")
            
            # Configure tesseract for Vietnamese
            custom_config = r'--oem 3 --psm 3 -l vie+eng'
            
            # Process with tesseract
            try:
                # Open with PIL
                with Image.open(filepath) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Extract text
                    text = pytesseract.image_to_string(img, config=custom_config)
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    
                    print(f"[OCR] Extracted {len(lines)} lines of text")
                    print(f"[OCR] Raw text:\n{text}")
                    
                    if not text.strip():
                        return jsonify({
                            'success': True,
                            'message': 'No text detected in image',
                            'full_text': '',
                            'lines': [],
                            'structured_data': {}
                        })
                    
                    # Extract structured data
                    structured_data = extract_structured_fields(text)
                    extracted_fields = {k: v for k, v in structured_data.items() if v}
                    
                    return jsonify({
                        'success': True,
                        'message': f'Successfully extracted {len(lines)} lines of text',
                        'full_text': text,
                        'lines': lines,
                        'structured_data': structured_data,
                        'extracted_fields': extracted_fields
                    })
                    
            except Exception as e:
                print(f"[OCR] Error processing image with tesseract: {str(e)}")
                return jsonify({'success': False, 'error': f'OCR processing error: {str(e)}'}), 500
                
        except Exception as e:
            print(f"[OCR] Error saving or opening image: {str(e)}")
            return jsonify({'success': False, 'error': f'Image handling error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[API] General error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced OCR Web App - Structured Field Extraction</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 0; }
        .container { max-width: 1200px; margin: 40px auto; background: white; border-radius: 12px; box-shadow: 0 2px 12px #0001; padding: 30px; }
        h1 { color: #3f51b5; margin-bottom: 10px; }
        .subtitle { color: #666; margin-bottom: 20px; font-size: 1.1em; }
        .upload-area { border: 2px dashed #bbb; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 20px; background: #f8f9fa; cursor: pointer; transition: border-color 0.2s; }
        .upload-area.dragover { border-color: #3f51b5; background: #e3f2fd; }
        .spinner { display: none; margin: 20px auto; border: 6px solid #f3f3f3; border-top: 6px solid #3f51b5; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
        .results { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-top: 20px; }
        .success { color: #2e7d32; }
        .error { color: #d32f2f; }
        .message { margin-bottom: 15px; font-weight: bold; }
        .tabs { display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 1em; transition: all 0.2s; }
        .tab.active { background: #3f51b5; color: white; border-radius: 6px 6px 0 0; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .structured-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .field-item { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; }
        .field-label { font-weight: bold; color: #3f51b5; margin-bottom: 5px; }
        .field-value { color: #333; word-break: break-word; }
        .field-empty { color: #999; font-style: italic; }
        pre { background: #e3f2fd; padding: 15px; border-radius: 6px; font-size: 0.9em; max-height: 400px; overflow-y: auto; }
        .stats { display: flex; justify-content: space-between; margin-bottom: 15px; }
        .stat-item { background: #e8f5e8; padding: 10px; border-radius: 6px; text-align: center; }
        .stat-number { font-size: 1.5em; font-weight: bold; color: #2e7d32; }
        .stat-label { font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enhanced OCR Web App</h1>
        <p class="subtitle">Upload Vietnamese vessel registration documents to extract structured key-value pairs</p>
        <div class="upload-area" id="uploadArea">
            <p>📄 Drag & drop a vessel document image here or click to select</p>
            <p style="font-size: 0.9em; color: #666;">Supports: PNG, JPG, JPEG, BMP, TIFF, WEBP</p>
            <input type="file" id="fileInput" accept="image/*" style="display:none;">
        </div>
        <div class="spinner" id="spinner"></div>
        <div class="results" id="results" style="display:none;">
            <div class="message" id="message"></div>
            <div class="stats" id="stats" style="display:none;">
                <div class="stat-item">
                    <div class="stat-number" id="textLinesCount">0</div>
                    <div class="stat-label">Text Lines</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="extractedFieldsCount">0</div>
                    <div class="stat-label">Extracted Fields</div>
                </div>
            </div>
            <div class="tabs">
                <button class="tab active" onclick="showTab('structured')">📋 Structured Data</button>
                <button class="tab" onclick="showTab('raw')">📝 Raw Text</button>
                <button class="tab" onclick="showTab('lines')">📄 Text Lines</button>
            </div>
            <div class="tab-content active" id="structured-tab">
                <div class="structured-grid" id="structuredGrid"></div>
            </div>
            <div class="tab-content" id="raw-tab">
                <h3>Full Text:</h3>
                <pre id="rawText"></pre>
            </div>
            <div class="tab-content" id="lines-tab">
                <h3>Text Lines (JSON Array):</h3>
                <pre id="linesJson"></pre>
            </div>
        </div>
    </div>
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const spinner = document.getElementById('spinner');
        const resultsDiv = document.getElementById('results');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', e => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', e => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        fileInput.addEventListener('change', e => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // Show selected tab
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }
        
        function handleFile(file) {
            resultsDiv.style.display = 'none';
            spinner.style.display = 'block';
            
            const formData = new FormData();
            formData.append('image', file);
            
            fetch('/api/ocr', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                spinner.style.display = 'none';
                resultsDiv.style.display = 'block';
                
                if (data.success) {
                    let messageClass = data.lines.length > 0 ? 'success' : 'error';
                    document.getElementById('message').innerHTML = `<span class="${messageClass}">${data.message}</span>`;
                    
                    // Update stats
                    document.getElementById('textLinesCount').textContent = data.lines.length;
                    document.getElementById('extractedFieldsCount').textContent = Object.keys(data.extracted_fields || {}).length;
                    document.getElementById('stats').style.display = 'flex';
                    
                    // Update structured data
                    updateStructuredData(data.structured_data || {});
                    
                    // Update raw text
                    document.getElementById('rawText').textContent = data.full_text || '';
                    
                    // Update lines JSON
                    document.getElementById('linesJson').textContent = JSON.stringify(data.lines, null, 2);
                    
                } else {
                    document.getElementById('message').innerHTML = `<span class="error">${data.error}</span>`;
                    document.getElementById('stats').style.display = 'none';
                }
            })
            .catch(err => {
                spinner.style.display = 'none';
                resultsDiv.style.display = 'block';
                document.getElementById('message').innerHTML = `<span class="error">Error: ${err.message}</span>`;
                document.getElementById('stats').style.display = 'none';
            });
        }
        
        function updateStructuredData(structuredData) {
            const grid = document.getElementById('structuredGrid');
            grid.innerHTML = '';
            
            const fieldLabels = {
                'certificate_number': 'Certificate Number',
                'vessel_type': 'Vessel Type',
                'gross_tonnage': 'Gross Tonnage',
                'dead_weight': 'Dead Weight',
                'length_overall': 'Length Overall',
                'length_design': 'Length Design',
                'breadth_overall': 'Breadth Overall',
                'breadth_design': 'Breadth Design',
                'depth': 'Depth',
                'draught': 'Draught',
                'hull_material': 'Hull Material',
                'construction_place': 'Construction Place',
                'engine_count': 'Engine Count',
                'engine_type': 'Engine Type',
                'engine_number': 'Engine Number',
                'total_power': 'Total Power',
                'power': 'Power',
                'engine_manufacture_place': 'Engine Manufacture Place',
                'port_registry': 'Port Registry',
                'registration_number': 'Registration Number',
                'registration_type': 'Registration Type',
                'issue_date': 'Issue Date',
                'issue_place': 'Issue Place',
                'valid_until': 'Valid Until'
            };
            
            Object.keys(fieldLabels).forEach(key => {
                const value = structuredData[key] || '';
                const fieldItem = document.createElement('div');
                fieldItem.className = 'field-item';
                fieldItem.innerHTML = `
                    <div class="field-label">${fieldLabels[key]}</div>
                    <div class="field-value ${value ? '' : 'field-empty'}">
                        ${value || 'Not detected'}
                    </div>
                `;
                grid.appendChild(fieldItem);
            });
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5011, debug=True) 
# 🐟 Fish Identification Project - Getting Started Guide

## Welcome to Fishial.ai Fish Identification System!

This project provides state-of-the-art fish identification using deep learning models. It can:

- **Detect fish** in images using YOLOv10
- **Classify fish species** from 426+ different species
- **Segment fish** with pixel-level accuracy
- **Process single images or batches**

## 📋 Prerequisites

- **Python 3.8+** (Check with `python3 --version`)
- **macOS, Linux, or Windows**
- **Basic terminal/command line knowledge**

## 🚀 Quick Start (5 Minutes!)

### 1. Set Up Environment

```bash
# Navigate to project directory
cd /path/to/fish-identification

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Basic Classification

```bash
# Run the simple classification example
python runner.py
```

This will classify the included `test_image.png` and show you the fish species!

### 3. Test YOLOv10 Fish Detection

```bash
# Go to YOLOv10 detector directory
cd detector_v10_m3

# Run fish detection on test image
python -c "
import sys
sys.path.append('.')
from inference import YOLOInference
import cv2

# Load detector
detector = YOLOInference('./model.ts', conf_threshold=0.25)

# Load and process image
image = cv2.imread('../test_image.png')
detections = detector.predict(image)

if detections and detections[0]:
    print(f'🎯 Found {len(detections[0])} fish!')
    for i, fish in enumerate(detections[0]):
        box = fish.get_box()
        score = fish.get_score()
        print(f'  Fish {i+1}: Box={box}, Confidence={score:.3f}')
else:
    print('❌ No fish detected')
"
```

### 4. Test Latest Classification Model

```bash
# Go to latest classification model directory
cd ../classification_rectangle_v7-1

# Run classification
python -c "
import sys
sys.path.append('.')
from inference import EmbeddingClassifier
import cv2

# Load classifier (426 species!)
classifier = EmbeddingClassifier('./model.ts', './database.pt')

# Classify test image
image = cv2.imread('../test_image.png')
results = classifier.inference_numpy(image)

if results:
    top_result = results[0]
    print(f\"🐠 Species: {top_result['name']}\")
    print(f\"🎯 Accuracy: {top_result['accuracy']:.3f}\")
    print(f\"🆔 Species ID: {top_result['species_id']}\")
else:
    print('❌ Could not classify fish')
"
```

## 📁 Project Structure

```
fish-identification/
├── 🤖 detector_v10_m3/              # YOLOv10 Fish Detector
│   ├── model.ts                     # TorchScript model (31MB)
│   └── inference.py                 # Detection script
├── 🔍 classification_rectangle_v7-1/ # Latest Fish Classifier
│   ├── model.ts                     # TorchScript model (108MB)
│   ├── database.pt                  # Species database (39MB)
│   ├── labels.json                  # 426 species labels
│   └── inference.py                 # Classification script
├── 🎭 segmentation models/          # Fish Segmentation
│   ├── segmentation_21_08_2023.ts   # TorchScript segmentation
│   └── segmentator_fpn_res18_416_1/ # ResNet18 segmentation
├── 📊 Legacy models/                # Older model versions
│   ├── classification_fishial_30_06_2023/
│   ├── final_cross_cross_entropy_*.ckpt
│   └── model_0259999.pth
├── 🛠️  Scripts & Tools/
│   ├── runner.py                    # Simple classification demo
│   ├── inference_class.py           # Legacy classifier
│   └── complete_fish_demo.py        # Full pipeline demo
├── 📋 requirements.txt              # Python dependencies
└── 🖼️  test_image.png              # Sample test image
```

## 🎯 Available Models

### 1. **YOLOv10 Fish Detector** (Latest - 2024)

- **Location**: `detector_v10_m3/`
- **Purpose**: Finds fish in images with bounding boxes
- **Input**: Any image with fish
- **Output**: Bounding boxes + confidence scores
- **Model size**: 31MB

### 2. **Fish Classifier v7-1** (Latest - 426 Species)

- **Location**: `classification_rectangle_v7-1/`
- **Purpose**: Identifies fish species using ConvNeXt Tiny backbone
- **Species**: 426 different fish species
- **Embedding size**: 128 dimensions
- **Model size**: 108MB

### 3. **Fish Segmentation Models**

- **MaskRCNN**: `segmentation_21_08_2023.ts` (213MB)
- **ResNet18 FPN**: `segmentator_fpn_res18_416_1/` (background/fish)
- **Purpose**: Pixel-level fish segmentation

## 💻 Usage Examples

### Basic Classification

```python
from inference_class import EmbeddingClassifier
import cv2

# Load model
classifier = EmbeddingClassifier(
    'model.ckpt',
    'embeddings.pt',
    'labels.json',
    'idx.json'
)

# Classify image
image = cv2.imread('your_fish_image.jpg')
result = classifier.inference_numpy(image)
print(f"Species: {result[0][0]}, Confidence: {result[0][1]}")
```

### YOLO Detection

```python
from detector_v10_m3.inference import YOLOInference
import cv2

# Load detector
detector = YOLOInference('./detector_v10_m3/model.ts')

# Detect fish
image = cv2.imread('your_image.jpg')
detections = detector.predict(image)

for fish in detections[0]:
    box = fish.get_box()
    confidence = fish.get_score()
    print(f"Fish at {box} with {confidence:.2f} confidence")
```

### Latest Classifier (v7-1)

```python
from classification_rectangle_v7-1.inference import EmbeddingClassifier
import cv2

# Load latest classifier
classifier = EmbeddingClassifier(
    './classification_rectangle_v7-1/model.ts',
    './classification_rectangle_v7-1/database.pt'
)

# Classify
image = cv2.imread('fish_image.jpg')
results = classifier.inference_numpy(image)

for result in results[:3]:  # Top 3 results
    print(f"Species: {result['name']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Species ID: {result['species_id']}")
```

## 🔗 Additional Resources

- **Live Demo**: [Fishial.ai Portal](https://portal.fishial.ai/search/by-fishial-recognition)
- **Google Colab**: [Interactive Notebook](https://colab.research.google.com/drive/1nKJ0V1sBLgfNJaCTQmuqUV1ybrx1m7qI?usp=sharing)
- **Website**: [www.fishial.ai](http://www.fishial.ai)
- **Model Downloads**: Check `README.md` for latest model links

## 🐛 Troubleshooting

### Common Issues:

1. **"externally-managed-environment" error**

   ```bash
   # Solution: Use virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Import errors**

   ```bash
   # Make sure you're in the virtual environment
   source venv/bin/activate

   # Install missing packages
   pip install opencv-python torch torchvision
   ```

3. **Model not found errors**

   ```bash
   # Check if model files exist
   ls -la detector_v10_m3/model.ts
   ls -la classification_rectangle_v7-1/model.ts
   ```

4. **Memory issues**
   - Use smaller batch sizes
   - Process images one at a time
   - Resize large images before processing

## 🎓 For Beginners

If you're new to Python/AI:

1. **Start simple**: Use `runner.py` first
2. **Understand the output**: Each model gives different information
3. **Try different images**: Test with various fish photos
4. **Read the code**: Look at `inference.py` files to understand how models work
5. **Experiment**: Try different confidence thresholds and parameters

## 📞 Getting Help

- Check existing issues in the project
- Read model documentation in each directory
- Test with the provided `test_image.png` first
- Make sure virtual environment is activated

**Happy Fish Identifying! 🐟🔍**

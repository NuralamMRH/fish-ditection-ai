from PIL import Image
import pytesseract
import sys

def process_image(image_path):
    try:
        # Open and preprocess the image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Configure tesseract parameters for better Vietnamese text recognition
        custom_config = r'--oem 3 --psm 3 -l vie+eng'
        
        # Get text using tesseract
        text = pytesseract.image_to_string(img, config=custom_config)
        
        print("\nRaw Text from Image:")
        print("=" * 50)
        print(text)
        print("=" * 50)
        
        return text
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "vessel_registration.jpg"
    
    process_image(image_path) 
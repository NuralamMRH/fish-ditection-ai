import base64
from PIL import Image
import io

# The image data from the conversation
image_data = '''
[Your vessel registration certificate image data will be here]
'''

# Remove header and convert to bytes
image_bytes = base64.b64decode(image_data)

# Create PIL Image
image = Image.open(io.BytesIO(image_bytes))

# Save the image
image.save('vessel_registration.jpg', 'JPEG') 
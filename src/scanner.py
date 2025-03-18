import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Check API key and model name
if not API_KEY:
    raise ValueError("❌ API_KEY not found in environment variables. Please check your .env file.")
if not MODEL_NAME:
    raise ValueError("❌ MODEL_NAME not found in environment variables. Please check your .env file.")

# Configure the API
genai.configure(api_key=API_KEY)

def extract_text_from_image(image_path, prompt):
    """Extract text from the image."""
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        print(f"🔍 Extracting text from {os.path.basename(image_path)}...")

        # Load image and send to the model
        image = Image.open(image_path)
        response = model.generate_content([image, prompt])

        return response.text if response else None
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return None

def process_image(image_path):
    """Process a single image and return the extracted text."""
    return extract_text_from_image(
        image_path, 
        "Extract the text in the image verbatim and correct any spelling mistakes if needed."
    )

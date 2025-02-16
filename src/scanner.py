import google.generativeai as genai
import os
import glob

API_KEY = 'AIzaSyDM9n_0ZTWWtlL0GX800mm59nw6Ee5TX8w'
if not API_KEY:
    raise ValueError("GEMINI_AI_API_KEY not available.")
genai.configure(api_key=API_KEY)

IMAGE_FOLDER = "answers"


def prep_image(image_path):
    """Uploads an image to Gemini AI and returns the file reference."""
    sample_file = genai.upload_file(
        path=image_path, display_name=os.path.basename(image_path))
    return sample_file


def extract_text_from_image(image_file, prompt):
    """Uses Gemini AI to extract text from an image."""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content([image_file, prompt])
    return response.text


def process_image(image_path):
    """Processes a single image and returns the extracted text."""
    sample_file = prep_image(image_path)
    text = extract_text_from_image(
        sample_file, "Extract the text in the image verbatim and correct any spelling mistakes if needed")
    return text

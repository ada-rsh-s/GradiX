import os
# Import the image processing function from scanner.py
from scanner import process_image
# Import the segregation function from segregator.py
from segregator import segregate_ocr_text


def main():
    # Path to the image you want to process
    # Replace with the actual path to your image
    image_path = "answers/sample_image.jpeg"

    # Step 1: Process the image using scanner.py to extract text
    extracted_text = process_image(image_path)

    # Step 2: Process the extracted OCR text using segregator.py
    if extracted_text:
        print(f"Extracted Text:\n{extracted_text}\n")
        # This will process the text and display the question-answer pairs
        segregate_ocr_text(extracted_text)
    else:
        print("Failed to extract text from the image.")


if __name__ == "__main__":
    main()

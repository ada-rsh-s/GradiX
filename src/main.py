import os
import glob
from scanner import process_image
from segregator import segregate_ocr_text
from tokenizer import preprocess_answers  # Tokenization & Lemmatization

IMAGE_FOLDER = "answers"

def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"⚠️ Folder '{IMAGE_FOLDER}' not found. Please check the path.")
        return
    
    image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not image_files:
        print(f"⚠️ No valid images found in folder '{IMAGE_FOLDER}'.")
        return

    print(f"\n🔍 Found {len(image_files)} valid images.\n")

    for i, image_file in enumerate(image_files, start=1):
        print(f"\n📸 Processing Image {i}/{len(image_files)}...")
        
        # Step 1: Extract text from image
        extracted_text = process_image(image_file)

        if extracted_text:
            print(f"\n📜 Extracted Text from {os.path.basename(image_file)}:\n{extracted_text}\n")
            
            # Step 2: Segregate into Q&A format
            qa_dict = segregate_ocr_text(extracted_text)

            # Step 3: Tokenize & Lemmatize answers
            processed_answers = preprocess_answers(qa_dict)

            # Step 4: Print final processed answers
            print("\n📝 Final Processed Answers:")
            for q, ans in processed_answers.items():
                print(f"Q{q}: {ans}")

        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()

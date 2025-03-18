import re

def segregate_ocr_text(ocr_text):
    """Splits OCR text into question-answer pairs."""
    question_pattern = re.compile(r'(\d{1,2}\s*\)\s*|\d{1,2}\s*\(\))')
    split_text = re.split(question_pattern, ocr_text)

    qa_dict = {}
    for i in range(1, len(split_text), 2):
        question_number = split_text[i].strip().replace(')', '').replace('(', '').replace(' ', '')
        answer = split_text[i + 1].strip()
        qa_dict[question_number] = answer

    return qa_dict  # Return dictionary instead of printing

import re
import pprint

def segregate_ocr_text(ocr_text):
    """Splits OCR text into question-answer pairs."""
    # Regular expression to match question numbers (with flexible formats like "1)", "1 )", "1))", etc.)
    question_pattern = re.compile(r'(\d{1,2}\s*\)\s*|\d{1,2}\s*\(\))')

    # Split the text based on question numbers
    split_text = re.split(question_pattern, ocr_text)

    # Initialize an empty dictionary to store questions and answers
    qa_dict = {}

    # Loop through the split text to populate the dictionary
    for i in range(1, len(split_text), 2):
        question_number = split_text[i].strip().replace(')', '').replace(
            '(', '').replace(' ', '')  # Remove extra characters
        answer = split_text[i + 1].strip()
        # Map the question number to its corresponding answer
        qa_dict[question_number] = answer


    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(qa_dict)

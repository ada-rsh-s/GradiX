from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
print("PaddleOCR installed successfully!")

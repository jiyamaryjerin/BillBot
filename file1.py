from paddleocr import PaddleOCR
from PIL import Image
import os


# 'en', 'ta', 'te', 'ka', 'devanagari'
lang_code = 'en'  
image_path = 'D:\\quantumX2025\\bills (handwritten)\\bill.jpg'  


ocr = PaddleOCR(use_angle_cls=True, lang=lang_code)

result = ocr.ocr(image_path, cls=True)

output_file = f'output_{lang_code}.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    if result:
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                confidence = word_info[1][1]
                f.write(f"{text} - Confidence: {confidence:.2f}\n")
        print(f"\nOCR complete. Output saved to: {os.path.abspath(output_file)}")
    else:
        f.write("No text detected.\n")
        print("\nNo text detected. Empty result written to file.")

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from IPython.display import display
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

def show_image(pathStr):
  img = Image.open(pathStr).convert("RGB")
  display(img)
  return img

def ocr_image(src_img):
  pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

hw_image = show_image("D:\\quantumX2025\\bills (handwritten)\\hand.jpg")

ocr_image(hw_image)

hw_image1 = hw_image.crop((0, 10, hw_image.size[0], 40))
#display(hw_image1)

print(ocr_image(hw_image1))

'''

image = Image.open("D:\\quantumX2025\\bills (handwritten)\\sis.jpg").convert("RGB")
image = image.resize((384, 384))
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Output:", text)
'''
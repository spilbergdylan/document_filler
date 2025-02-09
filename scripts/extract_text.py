import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from config import OUTPUT_DIR

def enhance_image(image):
    """Enhance image for better OCR performance."""
    print("🔍 DEBUG: Enhancing image for better OCR performance...")
    
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    print("✅ DEBUG: Image enhancement completed.")
    return processed

def extract_text_with_bboxes(image):
    """Extracts text with bounding boxes using Tesseract OCR."""
    print("🔍 DEBUG: Running Tesseract OCR...")

    extracted_data = []
    data = pytesseract.image_to_data(image, config="--psm 6 --oem 1", output_type=pytesseract.Output.DICT)

    if not data["text"]:  
        print("⚠️ WARNING: No text detected by Tesseract!")

    for i in range(len(data["text"])):
        if data["text"][i].strip():  # Ignore empty strings
            extracted_data.append({
                "text": data["text"][i],
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i]
            })
    
    print(f"✅ DEBUG: Extracted {len(extracted_data)} text entries from OCR.")
    return extracted_data

def extract_text_from_pdf(pdf_path):
    """Extracts text and bounding boxes from a PDF."""
    print(f"📄 DEBUG: Converting PDF '{pdf_path}' to images...")

    images = convert_from_path(pdf_path)
    if not images:
        print(f"🚨 ERROR: No images extracted from '{pdf_path}'! Check the PDF file.")
        return []

    extracted_results = []

    for page_num, image in enumerate(images, start=1):
        debug_page_path = os.path.join(OUTPUT_DIR, f"debug_page_{page_num}.png")
        image.save(debug_page_path)
        print(f"✅ DEBUG: Saved page {page_num} as image: {debug_page_path}")

        processed_image = enhance_image(image)
        extracted_data = extract_text_with_bboxes(processed_image)

        extracted_results.append({
            "page": page_num,
            "data": extracted_data
        })

    print(f"✅ DEBUG: OCR completed for {len(extracted_results)} pages.")
    return extracted_results

def extract_text(file_path):
    """Extracts text and bounding boxes from a PDF or an image."""
    print(f"📂 DEBUG: Processing file '{file_path}'...")

    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith((".png", "jpg", "jpeg", "tiff")):
        processed_image = enhance_image(Image.open(file_path))
        extracted_data = extract_text_with_bboxes(processed_image)

        print(f"✅ DEBUG: OCR completed for image '{file_path}'. Extracted {len(extracted_data)} text entries.")
        return extracted_data
    else:
        print("🚨 ERROR: Unsupported file format. Please use a PDF or an image.")
        raise ValueError("Unsupported file format. Please use a PDF or an image.")
    
if __name__ == "__main__":
# Example test file
    file_path = "data/one_page_test.pdf"

    print(f"📂 DEBUG: Running OCR on '{file_path}'...")

    extracted_data = extract_text(file_path)

    print("✅ DEBUG: Extracted Data:")
    print(extracted_data)

    # Save extracted data for debugging
    import json
    with open("output_debug.json", "w") as f:
        json.dump(extracted_data, f, indent=2)

    print("📂 DEBUG: Saved extracted data to 'output_debug.json'")


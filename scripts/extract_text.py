import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import json
from config import OUTPUT_DIR

def enhance_image(image):
    """Enhance image for better OCR performance."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.dilate(gray, kernel, iterations=1)  # Make text bolder
    return gray

def find_text_regions(image):
    """Use edge detection and contour detection to locate text areas."""
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 10:  # Ignore small noise
            bounding_boxes.append((x, y, w + 5, h + 5))  # Expand box slightly
    
    return bounding_boxes

def extract_text_with_bboxes(image):
    """Extracts text with bounding boxes using multiple Tesseract OCR PSM modes."""
    psm_modes = [11]  # Using PSM 11 for best field extraction
    best_data = []
    best_confidence = 0

    bounding_boxes = find_text_regions(image)
    for psm in psm_modes:
        custom_config = f"--oem 3 --psm {psm}"
        print(f"🔍 Running OCR with PSM mode: {psm}")
        extracted_data = []
        total_confidence = 0
        valid_words = 0

        for x, y, w, h in bounding_boxes:
            roi = image[y:y+h, x:x+w]  # Extract region of interest
            data = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data["text"])):
                if data["text"][i].strip():  # Ignore empty strings
                    confidence = int(data["conf"][i])
                    print(f"📌 Word: {data['text'][i]} | Confidence: {confidence}")
                    if confidence > 15:  # Lowered threshold to capture more text
                        total_confidence += confidence
                        valid_words += 1
                        extracted_data.append({
                            "text": data["text"][i],
                            "left": x + data["left"][i],
                            "top": y + data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i]
                        })
        
        avg_confidence = total_confidence / max(valid_words, 1)  # Avoid division by zero
        print(f"📊 PSM {psm} - Avg confidence: {avg_confidence:.2f}")
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_data = extracted_data
    
    # ✅ Sort text elements by Y (top) first, then X (left) for proper reading order
    best_data.sort(key=lambda entry: (entry["top"], entry["left"]))
    print(f"📊 Sorted {len(best_data)} extracted text elements")
    return best_data

def extract_text_from_pdf(pdf_path):
    """Extracts text and bounding boxes from a PDF."""
    print(f"📄 Processing PDF: {pdf_path}")
    images = convert_from_path(pdf_path)
    print(f"🔍 Extracted {len(images)} images from PDF")
    extracted_results = []

    for page_num, image in enumerate(images, start=1):
        print(f"📄 Processing page {page_num}...")
        processed_image = enhance_image(image)
        extracted_data = extract_text_with_bboxes(processed_image)
        extracted_results.append({"page": page_num, "data": extracted_data})
    return extracted_results

if __name__ == "__main__":
    pdf_path = "data/one_page_test.pdf"
    extracted_data = extract_text_from_pdf(pdf_path)

    extracted_data_path = os.path.join(OUTPUT_DIR, "extracted_text_with_bboxes_sorted.json")
    with open(extracted_data_path, "w") as f:
        json.dump(extracted_data, f, indent=2)

    print(f"📂 Data saved to {extracted_data_path}")
    print(f"🔍 Checking saved JSON content...")
    with open(extracted_data_path, "r") as f:
        print(f.read()[:500])  # Show first 500 characters

    print("✅ OCR extraction with sorted text order completed and saved.")

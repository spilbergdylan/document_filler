import os
import torch
from transformers import LayoutLMv3Processor
from PIL import Image
from pdf2image import convert_from_path
import json
from config import DATA_DIR, OUTPUT_DIR

# Load LayoutLMv3 Processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

def convert_pdf_to_images(pdf_path):
    """Convert a PDF to images and return the image paths."""
    images = convert_from_path(pdf_path)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths

def extract_text_blocks(image_path):
    """Extract text blocks with bounding boxes using LayoutLMv3 OCR."""
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Loaded Image Size: {image.size}")

    # Run OCR with LayoutLMv3
    encoding = processor(image, return_tensors="pt", return_offsets_mapping=True)
    ocr_tokens = encoding.tokens()

    # Flatten bounding boxes in case they're nested
    ocr_bboxes = encoding["bbox"].tolist()
    if isinstance(ocr_bboxes[0], list):  # Check if there's an extra nesting level
        ocr_bboxes = ocr_bboxes[0]

    print("🔍 OCR Tokens:", ocr_tokens)
    print("🔍 OCR Bounding Boxes:", ocr_bboxes)

    # Ensure OCR results exist
    if not ocr_tokens or not ocr_bboxes:
        print("🚨 Error: No OCR data extracted! Check input image quality.")
        return []

    # Group consecutive tokens into meaningful text blocks
    text_blocks = []
    current_text = ""
    current_bbox = None

    for i, (token, bbox) in enumerate(zip(ocr_tokens, ocr_bboxes)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue  # Skip special tokens

        token = token.replace("Ġ", " ").strip()  # Fix subword splitting

        if current_text:
            current_text += " " + token
            current_bbox[0] = min(current_bbox[0], bbox[0])  # Expand bbox
            current_bbox[1] = min(current_bbox[1], bbox[1])
            current_bbox[2] = max(current_bbox[2], bbox[2])
            current_bbox[3] = max(current_bbox[3], bbox[3])
        else:
            current_text = token
            current_bbox = bbox

        # Check spacing to determine if we should break
        if i + 1 < len(ocr_bboxes):
            next_bbox = ocr_bboxes[i + 1]
            horizontal_gap = abs(next_bbox[0] - bbox[2])

            # If gap is large, treat it as a new block
            if horizontal_gap > 50:
                text_blocks.append({
                    "text": current_text.strip(),
                    "bbox": current_bbox
                })
                current_text = ""
                current_bbox = None

    # Append last block
    if current_text:
        text_blocks.append({
            "text": current_text.strip(),
            "bbox": current_bbox
        })

    return text_blocks

if __name__ == "__main__":
    # Convert PDF to images
    pdf_path = os.path.join(DATA_DIR, "one_page_test.pdf")
    image_paths = convert_pdf_to_images(pdf_path)

    # Process each page
    all_text_blocks = []
    for image_path in image_paths:
        print(f"🔍 Processing: {image_path}")
        text_blocks = extract_text_blocks(image_path)
        all_text_blocks.extend(text_blocks)

    # Save extracted text and bounding boxes
    output_json = os.path.join(OUTPUT_DIR, "layoutlm_extracted_text.json")
    with open(output_json, "w") as f:
        json.dump(all_text_blocks, f, indent=2)

    print(f"📂 Extracted text blocks saved to {output_json}")

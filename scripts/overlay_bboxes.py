import os
import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from config import OUTPUT_DIR, DATA_DIR

# Load matched fields with bounding boxes
def load_matched_fields():
    matched_fields_path = os.path.join(OUTPUT_DIR, "matched_fields_with_bboxes.json")
    if not os.path.exists(matched_fields_path):
        print(f"❌ Error: {matched_fields_path} not found. Run field matching first.")
        return []
    with open(matched_fields_path, "r") as f:
        return json.load(f)

# Convert PDF to image
def load_image(file_path):
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path)
        if images:
            return images[0]  # Assume single-page PDF for now
        else:
            print(f"❌ Error: No images extracted from {file_path}.")
            return None
    return Image.open(file_path)

# Overlay bounding boxes on the image
def overlay_bboxes(image, matched_fields):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    for field in matched_fields:
        bbox = field.get("bbox")
        if bbox:
            x, y, w, h = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_cv, field["label"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image_cv

# Save the overlay image
def save_overlay_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"✅ Overlay image saved to {output_path}")

# Main execution
if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "one_page_test.pdf")  # Change if needed

    print("📄 Loading matched fields with bounding boxes...")
    matched_fields = load_matched_fields()

    print("🖼️ Loading original image...")
    image = load_image(file_path)
    if image is None:
        exit(1)

    print("🎨 Overlaying bounding boxes...")
    overlay_image = overlay_bboxes(image, matched_fields)

    output_path = os.path.join(OUTPUT_DIR, "overlay_image.png")
    save_overlay_image(overlay_image, output_path)

    print("✅ Process complete! Check the saved image for bounding boxes.")

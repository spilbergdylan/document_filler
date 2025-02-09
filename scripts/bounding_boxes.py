import os
import json
from fuzzywuzzy import process
from config import OUTPUT_DIR

# Load extracted text with bounding boxes
def load_extracted_text():
    extracted_text_path = os.path.join(OUTPUT_DIR, "extracted_data.json")
    if not os.path.exists(extracted_text_path):
        print(f"❌ Error: {extracted_text_path} not found. Run text extraction first.")
        return []
    with open(extracted_text_path, "r") as f:
        return json.load(f)

# Load extracted fields from GPT
def load_extracted_fields():
    extracted_fields_path = os.path.join(OUTPUT_DIR, "extracted_fields.json")
    if not os.path.exists(extracted_fields_path):
        print(f"❌ Error: {extracted_fields_path} not found. Run GPT field extraction first.")
        return []
    with open(extracted_fields_path, "r") as f:
        return json.load(f)

# Match GPT-extracted fields to OCR bounding boxes
def match_fields_to_bboxes(extracted_data, extracted_fields):
    matched_fields = []

    for field in extracted_fields:
        field_text = field["label"]  # The label GPT extracted
        best_match = None
        best_score = 0

        for entry in extracted_data:
            ocr_text = entry["text"]
            score = process.extractOne(field_text, [ocr_text])[1]  # Fuzzy match score

            if score > best_score:  # Find the closest match
                best_match = entry
                best_score = score

        if best_match and best_score > 80:  # Threshold for fuzzy matching
            field["bbox"] = best_match["bbox"]
        else:
            field["bbox"] = None  # No bounding box found

        matched_fields.append(field)

    return matched_fields

# Save matched fields to a new JSON file
def save_matched_fields(matched_fields):
    output_path = os.path.join(OUTPUT_DIR, "matched_fields_with_bboxes.json")
    with open(output_path, "w") as f:
        json.dump(matched_fields, f, indent=2)
    print(f"✅ Matched fields saved to {output_path}")

# Main execution
if __name__ == "__main__":
    print("📄 Loading extracted text with bounding boxes...")
    extracted_data = load_extracted_text()

    print("🤖 Loading extracted fields from GPT...")
    extracted_fields = load_extracted_fields()

    if not extracted_data or not extracted_fields:
        print("❌ Error: Missing data. Ensure both OCR and GPT extraction steps are completed.")
        exit(1)

    print("🔍 Matching fields to bounding boxes...")
    matched_fields = match_fields_to_bboxes(extracted_data, extracted_fields)

    print("💾 Saving matched fields with bounding boxes...")
    save_matched_fields(matched_fields)

    print("🎯 Matching complete!")

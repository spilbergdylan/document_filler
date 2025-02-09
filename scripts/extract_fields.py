import os
import json
from config import OUTPUT_DIR, DATA_DIR
from extract_text import extract_text
from extract_gpt import extract_fields_with_gpt

if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "one_page_test.pdf")

    print("📄 Processing file for text and bounding box extraction...")
    extracted_data = extract_text(file_path)

    if not extracted_data:
        print("⚠️ Warning: No text extracted from the document. Check OCR settings.")
        exit(1)

    # Save extracted text and bounding boxes
    raw_text = ""
    structured_data = []

    for page in extracted_data:
        page_num = page["page"]
        raw_text += f"\n\n--- Page {page_num} ---\n"
        for entry in page["data"]:
            raw_text += f"{entry['text']} "
            structured_data.append({
                "page": page_num,
                "text": entry["text"],
                "bbox": {
                    "left": entry["left"],
                    "top": entry["top"],
                    "width": entry["width"],
                    "height": entry["height"]
                }
            })

    raw_text = raw_text.strip()

    with open(os.path.join(OUTPUT_DIR, "raw_text.txt"), "w") as f:
        f.write(raw_text)

    with open(os.path.join(OUTPUT_DIR, "extracted_data.json"), "w") as f:
        json.dump(structured_data, f, indent=2)

    print("📜 Extracted Raw Text:")
    print(raw_text)

    print("🤖 Analyzing extracted text with GPT...")
    fields = extract_fields_with_gpt(raw_text)

    if not isinstance(fields, list):
        print("🚨 Error: GPT response is not a list. Please check the output format.")
        fields = []

    print("✅ Extracted Fields:")
    print(json.dumps(fields, indent=2))

    with open(os.path.join(OUTPUT_DIR, "extracted_fields.json"), "w") as f:
        json.dump(fields, f, indent=2)

    print("📂 Fields saved to output/extracted_fields.json")

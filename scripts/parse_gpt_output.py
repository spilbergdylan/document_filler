import os
import json
from config import OUTPUT_DIR

def parse_gpt_output(gpt_output):
    """Parses GPT's compressed output into structured JSON format."""
    structured_fields = []

    for line in gpt_output:
        parts = line.split(",")
        if len(parts) != 6:
            print(f"⚠️ Skipping invalid line: {line}")
            continue

        page, label, left, top, width, height = parts
        structured_fields.append({
            "page": int(page),
            "label": label.strip(),
            "bbox": {
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height)
            }
        })

    return structured_fields

if __name__ == "__main__":
    input_file = os.path.join(OUTPUT_DIR, "gpt_output.txt")
    output_file = os.path.join(OUTPUT_DIR, "parsed_fields.json")

    if not os.path.exists(input_file):
        print("❌ Error: GPT output file not found.")
        exit(1)

    with open(input_file, "r") as f:
        gpt_output = f.read().strip().split("\n")

    parsed_fields = parse_gpt_output(gpt_output)

    with open(output_file, "w") as f:
        json.dump(parsed_fields, f, indent=2)

    print(f"✅ Parsed fields saved to {output_file}")

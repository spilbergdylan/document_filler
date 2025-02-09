import json
from config import client, OUTPUT_DIR

def reformat_extracted_data(extracted_data):
    """Ensures bounding boxes are in the correct format."""
    formatted_data = []

    for page in extracted_data:
        page_num = page["page"]
        formatted_entries = []

        for entry in page["data"]:
            if "left" in entry and "top" in entry and "width" in entry and "height" in entry:
                entry["bbox"] = {
                    "left": entry.pop("left"),
                    "top": entry.pop("top"),
                    "width": entry.pop("width"),
                    "height": entry.pop("height"),
                }
            formatted_entries.append(entry)

        formatted_data.append({"page": page_num, "data": formatted_entries})

    return formatted_data


def extract_fields_with_gpt(extracted_data):
    """Uses GPT to analyze structured text with bounding boxes and return only form fields, merging related words dynamically."""

    # Ensure bounding boxes are structured correctly
    extracted_data = reformat_extracted_data(extracted_data)

    formatted_lines = []
    for page in extracted_data:
        page_num = page["page"]
        for entry in page["data"]:
            text = entry["text"].replace(",", "")  # Remove commas to prevent CSV issues
            bbox = entry["bbox"]
            formatted_lines.append(f"{page_num},{text},{bbox['left']},{bbox['top']},{bbox['width']},{bbox['height']}")

    formatted_text = "\n".join(formatted_lines)

    prompt = f"""
    You are an expert in form analysis. The following is extracted text from a document, including bounding box locations.
    Your task is to:
    1. Identify fields that need to be filled.
    2. **Merge words that belong to the same field** (e.g., "Federal", "Tax", "ID", "Number" should become "Federal Tax ID Number").
    3. **Maintain bounding box accuracy** by adjusting width/height to fit merged fields.
    4. **Only return the merged form fields** in the same structured format.

    ### **Input format** (CSV-like):
    PageNumber,Label,X,Y,Width,Height

    ### **Example input**:
    ```
    1,First,100,150,40,20
    1,Name,145,150,60,20
    1,Date,200,150,50,20
    1,of,255,150,20,20
    1,Birth,280,150,60,20
    1,Signature,300,400,120,30
    ```

    ### **Expected output**:
    ```
    1,First Name,100,150,105,20
    1,Date of Birth,200,150,140,20
    1,Signature,300,400,120,30
    ```

    **Here is the extracted data:**
    ```
    {formatted_text}
    ```

    **Return only the relevant merged form fields in the exact format shown above. Do not add any explanations.**
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
        )

        # Get GPT's raw response
        response_text = chat_completion.choices[0].message.content.strip()
        print("🔍 GPT Raw Response:", response_text)

        return response_text.split("\n")  # Return as a list of CSV-like rows

    except Exception as e:
        print("🚨 Error calling GPT:", e)
        return []

if __name__ == "__main__":
    extracted_data_path = f"{OUTPUT_DIR}/extracted_text_with_bboxes.json"

    print("📄 Loading extracted data with bounding boxes...")
    with open(extracted_data_path, "r") as f:
        extracted_data = json.load(f)

    print("🔄 Formatting extracted data to ensure bbox structure...")
    extracted_data = reformat_extracted_data(extracted_data)  # Fix bbox format

    print("🤖 Sending optimized data to GPT for field merging...")
    gpt_output = extract_fields_with_gpt(extracted_data)

    gpt_output_path = f"{OUTPUT_DIR}/gpt_output.txt"
    with open(gpt_output_path, "w") as f:
        f.write("\n".join(gpt_output))

    print(f"✅ GPT output with merged fields saved to {gpt_output_path}")

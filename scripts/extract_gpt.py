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
    You are an expert in document form analysis and field identification. Your task is to analyze the following OCR-extracted text with bounding box locations and identify the form fields that need to be filled. Apply intelligent merging and contextual understanding to reconstruct the fields accurately while ensuring the text remains in reading order.

    ### Key Responsibilities:
    1. Identify form fields from the extracted text.
       - Fields include names, dates, addresses, telephone numbers, checkboxes, and identification numbers.
       - Ignore non-field text (headers, instructions, labels unrelated to fields).
    
    2. Merge multi-word fields when necessary.
       - Example: "First", "Name" → "First Name"
       - "Date", "of", "Birth" → "Date of Birth"
       - "Federal", "Tax", "ID", "Number" → "Federal Tax ID Number"
    
    3. Maintain bounding box accuracy while merging fields.
       - When merging words, extend the bounding box width to cover the entire phrase.
       - The Y position (top) should remain the same to preserve alignment.
    
    4. Contextual labeling:
       - If a Telephone Number is next to "Office Manager," label it as "Office Manager Telephone Number".
       - If a Name is linked to a department, assign a contextual label like "Department Head Name".
       - Ensure differentiation between fields such as "Office City", "Office Phone Number", and "Personal Telephone Number" based on contextual placement.
       
    5. Contextual Field Labeling:
       - Based on the surrounding text and field placement, determine the specific context of the field.
       - Example: If a phone number field is near "Primary Office," label it as "Primary Office Phone Number".
       - Ensure the label accurately reflects the field's purpose and context within the document.
       - if it is any number or field like a fax number for example, label it according to whos fax number it is.

    ### Input Format (CSV-like):
    PageNumber,Label,X,Y,Width,Height

    ### Example Input:
    ```
    1,First,100,150,40,20
    1,Name,145,150,60,20
    1,Date,200,150,50,20
    1,of,255,150,20,20
    1,Birth,280,150,60,20
    1,Signature,300,400,120,30
    ```
    
    ### Expected Output:
    ```
    1,First Name,100,150,105,20
    1,Date of Birth,200,150,140,20
    1,Signature,300,400,120,30
    ```
    
    ### Here is the extracted OCR data from the document:
    ```
    {formatted_text}
    ```

    **ONLY return the merged form fields in the structured format above. Do NOT include explanations, headers, or unrelated text. Ensure accurate bounding box adjustments for merged fields. Maintain reading order to preserve document structure.**
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
        )

        response_text = chat_completion.choices[0].message.content.strip()
        print("🔍 GPT Raw Response:", response_text)
        return response_text.split("\n")
    
    except Exception as e:
        print("🚨 Error calling GPT:", e)
        return []

if __name__ == "__main__":
    extracted_data_path = f"{OUTPUT_DIR}/extracted_text_with_bboxes_sorted.json"

    print("📄 Loading extracted data with bounding boxes...")
    with open(extracted_data_path, "r") as f:
        extracted_data = json.load(f)

    print("🔄 Formatting extracted data to ensure bbox structure...")
    extracted_data = reformat_extracted_data(extracted_data)

    print("🤖 Sending optimized data to GPT for field merging...")
    gpt_output = extract_fields_with_gpt(extracted_data)

    gpt_output_path = f"{OUTPUT_DIR}/gpt_output.txt"
    with open(gpt_output_path, "w") as f:
        f.write("\n".join(gpt_output))

    print(f"✅ GPT output with merged fields saved to {gpt_output_path}")
import cv2
import json

def fill_form(image_path, mapped_data_path, signature_path, output_path):
    """Fills the form with mapped data, including dynamic date and signature handling."""
    # Load the form image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from {image_path}")

    # Load mapped data
    with open(mapped_data_path, "r") as f:
        mapped_data = json.load(f)

    # Load the doctor's signature
    signature = cv2.imread(signature_path, cv2.IMREAD_UNCHANGED)
    if signature is None:
        raise FileNotFoundError(f"Failed to load signature image from {signature_path}")

    for field in mapped_data:
        field_type = field.get("type")
        label = field.get("label", "")
        value = field.get("value", "")

        if value == "SIGNATURE" and field_type == "line":  # Insert the signature
            x1, y1, x2, y2 = field["x1"], field["y1"], field["x2"], field["y2"]
            signature_resized = cv2.resize(signature, (x2 - x1, y2 - y1))

            # Overlay signature if transparent, else paste it directly
            if signature_resized.shape[2] == 4:  # Check for alpha channel
                overlay_image(image, signature_resized, x1, y1)
            else:
                image[y1:y1 + signature_resized.shape[0], x1:x1 + signature_resized.shape[1]] = signature_resized

        elif field_type == "line":  # Write text values on the lines
            x1, y1, _, _ = field["x1"], field["y1"], field["x2"], field["y2"]
            position = (x1 + 5, y1 - 5)  # Adjust text position slightly above the line
            cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        elif field_type == "box":  # Handle checkboxes
            if value.lower() in ["yes", "true", "checked", "accept"]:  # If checkbox is marked
                x1, y1, x2, y2 = field["x1"], field["y1"], field["x2"], field["y2"]
                # Draw a checkmark inside the box
                cv2.line(image, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (0, 0, 0), 2)
                cv2.line(image, (x1 + 5, y2 - 5), (x2 - 5, y1 + 5), (0, 0, 0), 2)

    # Save the filled form
    cv2.imwrite(output_path, image)
    print(f"Filled form saved to {output_path}")


def overlay_image(background, overlay, x, y):
    """Overlays a transparent image on top of a background image."""
    for i in range(overlay.shape[0]):
        for j in range(overlay.shape[1]):
            if overlay[i, j][3] > 0:  # Alpha channel > 0
                background[y + i, x + j] = overlay[i, j][:3]  # Copy BGR values


if __name__ == "__main__":
    # Paths for input files
    image_path = r"C:\Users\spilb\document_filler\data\hepb_sample_form.png"
    mapped_data_path = r"C:\Users\spilb\document_filler\output\mapped_data.json"
    signature_path = r"C:\Users\spilb\document_filler\data\doctor_signature.png"
    output_path = r"C:\Users\spilb\document_filler\output\filled_form.png"

    # Fill the form
    fill_form(image_path, mapped_data_path, signature_path, output_path)

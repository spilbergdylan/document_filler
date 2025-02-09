import os
from openai import OpenAI
import pytesseract

# 🔹 OpenAI Configuration
OPENAI_API_KEY = os.getenv("s")  # Load API key from environment variable

client = OpenAI(api_key="test")

# 🔹 Tesseract Configuration (Set Path for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../output")

# 🔹 Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

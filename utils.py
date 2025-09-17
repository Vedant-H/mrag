import hashlib
import io
import os
import sys
import google.generativeai as genai
import numpy as np
import fitz
from PIL import Image


def get_text_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Generates an embedding for the given text using Google's text-embedding-004 model."""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type
        )
        return np.array(response['embedding'])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

# def load_pdf_content(pdf_path: str) -> dict:
#     """
#     Extracts text from each page and saves images from each page of a PDF file.
#     Returns a dictionary containing the full document text, its unique ID,
#     and a list of page-level data (text, associated image paths).
#     """
#     full_document_text = ""
#     pages_data = []
    
#     # Create a unique ID for the document using its content hash for the temp image folder
#     with open(pdf_path, 'rb') as f:
#         pdf_content = f.read()
#     doc_unique_id = hashlib.md5(pdf_content).hexdigest()
#     images_dir = os.path.join("temp_pdf_images", doc_unique_id)
#     os.makedirs(images_dir, exist_ok=True) # Ensure the directory exists

#     print(f"Extracting content from: {pdf_path}")
#     try:
#         document = fitz.open(pdf_path)
#         for page_num in range(len(document)):
#             page = document.load_page(page_num)
#             page_text = page.get_text()
#             full_document_text += page_text + "\n" # Accumulate full text

#             page_image_paths = []
#             # Extract images from the page
#             img_list = page.get_images(full=True)
#             for img_index, img in enumerate(img_list):
#                 xref = img[0]
#                 base_image = document.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]

#                 # Fallback to PNG if extension is not recognized or not standard
#                 if image_ext.lower() not in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']:
#                     image_ext = 'png' 

#                 image_filename = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
#                 image_path = os.path.join(images_dir, image_filename)

#                 try:
#                     # Use PIL to save the image bytes to a file
#                     Image.open(io.BytesIO(image_bytes)).save(image_path)
#                     page_image_paths.append(image_path)
#                 except Exception as e:
#                     print(f"Warning: Could not save image {image_filename} from page {page_num+1}. Error: {e}")
#                     # Continue processing other images/pages
            
#             pages_data.append({
#                 "page_num": page_num, # 0-indexed page number
#                 "text": page_text,
#                 "image_paths": page_image_paths
#             })
        
#         print(f"Successfully extracted text and images to '{images_dir}' for {pdf_path}")
#         return {
#             "document_id": doc_unique_id,
#             "full_text": full_document_text,
#             "pages_data": pages_data # List of dicts, each with page_num, text, image_paths
#         }

#     except fitz.FileDataError:
#         print(f"Error: Could not open or read PDF file at {pdf_path}. It might be corrupted or not a valid PDF.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"An unexpected error occurred while reading PDF: {e}")
#         sys.exit(1)


import os
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def ask_gemini_about_image(image_path, prompt):
    """Send an image + prompt to Gemini and return the response text."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    with open(image_path, "rb") as f:
        img_data = f.read()
    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": img_data}]
    )
    return response.text if response and response.text else ""


def extract_text_tables(page):
    """
    Try to detect text-based tables from PDF page text.
    Returns list of tables in CSV format.
    """
    tables = []
    blocks = page.get_text("blocks")  # get structured blocks of text
    
    # A naive approach: detect blocks with multiple spaces that look tabular
    for b in blocks:
        text = b[4]
        if "\t" in text or "  " in text:  # multiple spaces or tabs
            # Try to normalize rows into CSV
            rows = []
            for line in text.splitlines():
                # split on whitespace and join with commas
                row = ",".join(line.split())
                rows.append(row)
            if rows:
                tables.append("\n".join(rows))
    
    return tables



def load_pdf_content(pdf_path, doc_unique_id="default_doc"):
    """
    Extracts text, images, tables, and charts from PDF.
    Uses Gemini to classify and interpret tables/charts.
    Returns structured data for ChromaDB or downstream use.
    """
    doc = fitz.open(pdf_path)
    all_text = ""
    pages_data = []

    os.makedirs("temp_pdf_images", exist_ok=True)

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        all_text += page_text + "\n"

        text_tables = extract_text_tables(page)

        image_paths = []   # 👈 this matches your old pipeline
        tables_info = []
        charts_info = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            img_path = f"temp_pdf_images/page{page_num}_img{img_index}.png"
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Step 1: Classify image
            classification_prompt = (
                "Classify this image strictly as one of: TABLE, CHART, OTHER."
            )
            classification = ask_gemini_about_image(img_path, classification_prompt).strip().upper()

            if "TABLE" in classification:
                # Step 2: Extract table as CSV
                table_prompt = "Extract this table into CSV format. Use commas to separate columns."
                table_csv = ask_gemini_about_image(img_path, table_prompt)
                tables_info.append({"image_path": img_path, "table_csv": table_csv})

            elif "CHART" in classification:
                # Step 2: Generate chart caption/summary
                chart_prompt = "Describe this chart in detail. Include axes, data trend, and summary."
                chart_caption = ask_gemini_about_image(img_path, chart_prompt)
                charts_info.append({"image_path": img_path, "chart_caption": chart_caption})

            else:
                # Store as generic image
                image_paths.append(img_path)

        pages_data.append({
            "page_num": page_num,
            "text": page_text,
            "image_paths": image_paths,  # 👈 restored old key
            "tables": tables_info + text_tables,
            "charts": charts_info
        })

    return {
        "document_id": doc_unique_id,
        "full_text": all_text,
        "pages_data": pages_data
    }

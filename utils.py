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

def load_pdf_content(pdf_path: str) -> dict:
    """
    Extracts text from each page and saves images from each page of a PDF file.
    Returns a dictionary containing the full document text, its unique ID,
    and a list of page-level data (text, associated image paths).
    """
    full_document_text = ""
    pages_data = []
    
    # Create a unique ID for the document using its content hash for the temp image folder
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    doc_unique_id = hashlib.md5(pdf_content).hexdigest()
    images_dir = os.path.join("temp_pdf_images", doc_unique_id)
    os.makedirs(images_dir, exist_ok=True) # Ensure the directory exists

    print(f"Extracting content from: {pdf_path}")
    try:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_text = page.get_text()
            full_document_text += page_text + "\n" # Accumulate full text

            page_image_paths = []
            # Extract images from the page
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Fallback to PNG if extension is not recognized or not standard
                if image_ext.lower() not in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']:
                    image_ext = 'png' 

                image_filename = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                image_path = os.path.join(images_dir, image_filename)

                try:
                    # Use PIL to save the image bytes to a file
                    Image.open(io.BytesIO(image_bytes)).save(image_path)
                    page_image_paths.append(image_path)
                except Exception as e:
                    print(f"Warning: Could not save image {image_filename} from page {page_num+1}. Error: {e}")
                    # Continue processing other images/pages
            
            pages_data.append({
                "page_num": page_num, # 0-indexed page number
                "text": page_text,
                "image_paths": page_image_paths
            })
        
        print(f"Successfully extracted text and images to '{images_dir}' for {pdf_path}")
        return {
            "document_id": doc_unique_id,
            "full_text": full_document_text,
            "pages_data": pages_data # List of dicts, each with page_num, text, image_paths
        }

    except fitz.FileDataError:
        print(f"Error: Could not open or read PDF file at {pdf_path}. It might be corrupted or not a valid PDF.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading PDF: {e}")
        sys.exit(1)

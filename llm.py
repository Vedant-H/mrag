import os
import re
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

async def generate_answer_with_llm(
    question: str,
    retrieved_context: str,
    chat_history: list,
    retrieved_image_paths: list = None
) -> str:
    """
    Generates an answer using Gemini 1.5 Flash with text + image input.
    Handles question, context, chat history, and optional images.
    """

    try:
        genai_model = genai.GenerativeModel("gemini-1.5-flash")

        # ---------- Build Prompt Parts ----------
        parts = []

        # System instruction
        parts.append(
            "You are a helpful and precise assistant. Use the following context and images to answer the question.\n"
            "Use your general knowledge only if the answer isn't available in the provided data.\n"
            "If you use information from an image, explicitly mention the image by its filename and page number, e.g., 'As seen in Image: page_6_img_1.png (Page 6), ...' or 'Referring to Image: page_4_img_1.png (Page 4) which is a flowchart, ...'."
        )

        # # Add chat history (keep this before context for better conversational flow)
        # history_str = ""
        # if chat_history:
        #     history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
        #     history_str = "--- Chat History ---\n" + history_str + "\n--------------------"
        #     parts.append(history_str)

        # Add context if available
        if retrieved_context:
            parts.append(f"\nContext:\n{retrieved_context}")

        # Add images (if provided)
        if retrieved_image_paths:
            parts.append("\n--- Relevant Images ---")
            for path in retrieved_image_paths:
                try:
                    if not os.path.exists(path):
                        print(f"[Warning] Image not found: {path}")
                        continue

                    image = Image.open(path).convert("RGB")
                    parts.append(image)

                    # Optional caption/metadata - IMPORTANT FOR LLM REFERENCE
                    filename = os.path.basename(path)
                    # Extract page number more robustly: 'page_X_img_Y.ext' -> X
                    page_match = re.search(r'page_(\d+)_img', filename) # Using regex for more robust extraction
                    page_num_str = page_match.group(1) if page_match else "unknown"
                    parts.append(f"Image: {filename} (Page {page_num_str})")

                except Exception as err:
                    print(f"[Warning] Failed to process image {path}: {err}")
            parts.append("-----------------------")

        # Final question
        parts.append(f"\nQuestion:\n{question}\n\nAnswer:")

        # ---------- Call Gemini ----------
        response = await genai_model.generate_content_async(parts)
        return response.text

    except Exception as e:
        print(f"[Error] LLM generation failed: {e}")
        return "Sorry, I couldn't generate an answer due to an internal error."

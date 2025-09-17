# import os
# import google.generativeai as genai
# from PIL import Image # For loading image files
# import asyncio # For running async functions
# GOOGLE_API_KEY =
# genai.configure(api_key=GOOGLE_API_KEY)

# # --- Function to load an image ---
# def load_image(image_path: str) -> Image.Image:
#     """Loads an image from the specified path using Pillow."""
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image file not found: {image_path}")
    
#     try:
#         # Open image and convert to RGB (some models prefer this)
#         img = Image.open(image_path).convert("RGB")
#         print(f"Successfully loaded image from: {image_path}")
#         return img
#     except Exception as e:
#         raise ValueError(f"Could not load or process image from {image_path}: {e}")

# # --- Async function to interact with Gemini Multimodal ---
# async def describe_image_with_gemini(image_path: str, user_prompt: str = "") -> str:
#     """
#     Sends an image and an optional text prompt to Gemini 1.5 Flash
#     and returns its textual description/analysis.
#     """
#     try:
#         # Load the image
#         img = load_image(image_path)

#         # Get the generative model
#         model = genai.GenerativeModel("gemini-1.5-flash")

#         # Prepare the content list for the multimodal input
#         # The order matters: instruction text first, then image, then specific question
#         contents_for_gemini = [f"You are an AI assistant that describes images.Analyze the provided image and respond to the user's request. Be detailed and insightful. {user_prompt}"                
#         ,img ]
        
#         print("\nSending image and prompt to Gemini 1.5 Flash...")
#         response = await model.generate_content_async(contents_for_gemini)
        
#         return response.text

#     except Exception as e:
#         print(f"Error during Gemini interaction: {e}")
#         return "Sorry, I couldn't process the image due to an error."

# # --- Main function to run the test ---
# async def main():
#     image_file_path = input("Enter the path to an image file (e.g., 'path/to/my_image.jpg'): ")
#     user_request = input("Enter a specific request for the image (optional, press Enter to skip): ")

#     if not os.path.exists(image_file_path):
#         print(f"Error: The file '{image_file_path}' does not exist. Please check the path and try again.")
#         return

#     print(f"\nTesting image input with Gemini for: {image_file_path}")
#     result = await describe_image_with_gemini(image_file_path, user_request)

#     print("\n--- Gemini's Response ---")
#     print(result)
#     print("-------------------------")

# # --- Run the main function ---
# if __name__ == "__main__":
#     asyncio.run(main())

# C:\Users\Vedant\Downloads\test2.pdf


# --- LLM Generation Function ---
# async def generate_answer_with_llm(question: str, retrieved_context: str, chat_history: list, retrieved_image_paths: list = None) -> str:
#     """
#     Generates an answer using Google Gemini 1.5 Flash based on the question, retrieved context,
#     previous chat history, and retrieved images.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")

#         history_str = ""
#         if chat_history:
#             history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
#             history_str = "--- Chat History ---\n" + history_str + "\n--------------------"

#         # Prepare multimodal content for Gemini
#         # Gemini's generate_content expects a list of parts (Text, Image)
#         contents_for_gemini = []

#         # Add the main RAG prompt instruction
#         contents_for_gemini.append(genai.types.TextPart(
#             "You are a helpful and precise assistant for question-answering. "
#             "Prioritize answering based on the following context. "
#             "If the answer cannot be found in the provided context (text or images), you may use your general knowledge "
#             "to provide a concise and relevant answer. If images are provided, refer to them explicitly if needed for the answer. "
#             "Be descriptive and insightful."
#         ))

#         # Add chat history
#         if history_str:
#             contents_for_gemini.append(genai.types.TextPart(history_str))
        
#         # Add retrieved text context
#         if retrieved_context:
#             contents_for_gemini.append(genai.types.TextPart("\nContext:\n" + retrieved_context))

#         # Add retrieved images
#         if retrieved_image_paths:
#             contents_for_gemini.append(genai.types.TextPart("\n--- Relevant Images ---"))
#             for img_path in retrieved_image_paths:
#                 try:
#                     # Check if image file exists before trying to open
#                     if not os.path.exists(img_path):
#                         print(f"Warning: Image file not found at {img_path}. Skipping.")
#                         continue

#                     # Open image and convert to RGB (required by some models)
#                     img = Image.open(img_path).convert("RGB")
#                     contents_for_gemini.append(genai.types.ImagePart(img))
#                     # Optional: Add a text part to describe the image's source/filename to the LLM
#                     contents_for_gemini.append(genai.types.TextPart(f"Image from path: {os.path.basename(img_path)} (Page {os.path.basename(img_path).split('_')[1]}).\n"))
#                 except Exception as e:
#                     print(f"Warning: Could not load or process image from {img_path} for LLM: {e}")
#             contents_for_gemini.append(genai.types.TextPart("-----------------------"))

#         # Add the question
#         contents_for_gemini.append(genai.types.TextPart("\nQuestion:\n" + question + "\n\nAnswer:"))

#         response = await model.generate_content_async(contents_for_gemini)
#         return response.text
#     except Exception as e:
#         print(f"Error generating answer with LLM: {e}")
#         return "Sorry, I couldn't generate an answer due to an internal error."

# working
# async def generate_answer_with_llm(
#     question: str,
#     retrieved_context: str,
#     chat_history: list,
#     retrieved_image_paths: list = None
# ) -> str:
#     """
#     Generates an answer using Gemini 1.5 Flash with text + image input.
#     Handles question, context, chat history, and optional images.
#     """

#     try:
#         genai_model = genai.GenerativeModel("gemini-1.5-flash")

#         # ---------- Build Prompt Parts ----------
#         parts = []

#         # System instruction
#         parts.append(
#             "You are a helpful and precise assistant. Use the following context and images to answer the question.\n"
#             "Use your general knowledge only if the answer isn't available in the provided data."
#         )

#         # Add context if available
#         if retrieved_context:
#             parts.append(f"\nContext:\n{retrieved_context}")

#         # Add images (if provided)
#         if retrieved_image_paths:
#             parts.append("\n--- Relevant Images ---")
#             for path in retrieved_image_paths:
#                 try:
#                     if not os.path.exists(path):
#                         print(f"[Warning] Image not found: {path}")
#                         continue

#                     image = Image.open(path).convert("RGB")
#                     parts.append(image)

#                     # Optional caption/metadata
#                     filename = os.path.basename(path)
#                     page = filename.split("_")[1] if "_" in filename else "unknown"
#                     parts.append(f"Image: {filename} (Page {page})")

#                 except Exception as err:
#                     print(f"[Warning] Failed to process image {path}: {err}")
#             parts.append("-----------------------")

#         # Final question
#         parts.append(f"\nQuestion:\n{question}\n\nAnswer:")

#         # ---------- Call Gemini ----------
#         response = await genai_model.generate_content_async(parts)
#         return response.text

#     except Exception as e:
#         print(f"[Error] LLM generation failed: {e}")
#         return "Sorry, I couldn't generate an answer due to an internal error."


# In your `generate_answer_with_llm` function:

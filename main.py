import re
import shutil 
import asyncio 
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from llm import generate_answer_with_llm
from data import get_db_connection, store_chunk_in_db , create_table_if_not_exists, retrieve_top_k_chunks , clear_all_chunks_from_db
from utils import get_text_embedding, load_pdf_content 
from chro import store_chunk_in_chroma, retrieve_top_k_chunks_from_chroma

async def main():
    conn = None
    doc_unique_id = None # Initialize for cleanup
    images_dir = None # Initialize for cleanup

    try:
        # 1. Connect to the database
        # conn = get_db_connection()

        # # 2. Ensure the table exists and HNSW index is created
        # create_table_if_not_exists(conn)

        # 3. Prompt user for PDF path
        pdf_path = input("Enter the path to your PDF file (e.g., 'document.pdf'): ")
        if not os.path.exists(pdf_path):
            print(f"Error: File not found at {pdf_path}")
            sys.exit(1)
        if not pdf_path.lower().endswith(".pdf"):
            print(f"Error: The provided file is not a PDF. Please provide a .pdf file.")
            sys.exit(1)

        # 4. Clear existing data and associated temporary images from previous runs
        # clear_all_chunks_from_db(conn)
        if os.path.exists("temp_pdf_images"):
            try:
                # Remove all previous document image folders
                for item in os.listdir("temp_pdf_images"):
                    item_path = os.path.join("temp_pdf_images", item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print("Cleaned up existing temporary image directories.")
            except Exception as e:
                print(f"Error cleaning up old image directories: {e}")


        # 5. Load text and images from the new PDF
        pdf_content_data = load_pdf_content(pdf_path)
        doc_unique_id = pdf_content_data["document_id"]
        document_text = pdf_content_data["full_text"] # The full text is not used directly for chunking now
        pages_data = pdf_content_data["pages_data"]
        images_dir = os.path.join("temp_pdf_images", doc_unique_id) # Set path for cleanup in finally block

        if not document_text.strip() and not any(p['image_paths'] for p in pages_data):
            print("Error: Extracted text and images from PDF are empty. Cannot process.")
            sys.exit(1)

        document_filename = os.path.basename(pdf_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        
        print(f"\nProcessing chunks for document '{document_filename}' (ID: {doc_unique_id[:8]}...)...")
        total_chunks_processed = 0
        # Iterate through page-level data to create chunks
        for page_data in pages_data:
            page_text = page_data["text"]
            page_number = page_data["page_num"] # 0-indexed page number
            page_image_paths = page_data["image_paths"] # Images extracted from this specific page

            if page_text.strip():
                page_text_chunks = text_splitter.split_text(page_text)
                for chunk_text in page_text_chunks:
                    embedding = get_text_embedding(chunk_text, task_type="RETRIEVAL_DOCUMENT")
                    store_chunk_in_chroma(doc_unique_id, chunk_text, embedding, page_image_paths, page_number)
                    total_chunks_processed += 1
            elif page_image_paths:
                dummy_text = f"This chunk represents content from page {page_number+1} which contains images but no discernible text. The images might contain diagrams, graphs, or visual information related to the document."
                embedding = get_text_embedding(dummy_text, task_type="RETRIEVAL_DOCUMENT")
                store_chunk_in_chroma(doc_unique_id, chunk_text, embedding, page_image_paths, page_number)
                total_chunks_processed += 1
        print(f"All {total_chunks_processed} chunks processed and stored in the database.")


        # 6. Start the conversational loop
        chat_history = []
        MAX_HISTORY_TURNS = 0# Store previous 3 user-AI turns

        query_text = input("\nWrite a query about the document (type 'exit' to quit): ")

        while query_text.lower() != 'exit':
            # --- 6.1. Incorporate history for Query Embedding (Query Expansion) ---
            conversational_query_parts = []
            for q, a in chat_history[-MAX_HISTORY_TURNS:]:
                conversational_query_parts.append(f"Q: {q}\nAI: {a}") # Add newline after Q/A
            conversational_query_parts.append(f"Current Q: {query_text}")

            conversational_query = "\n".join(conversational_query_parts) # Join with newline for clarity
            print(f"\nEmbedding query with context:\n'{conversational_query}'") # Better formatting

            query_embedding = get_text_embedding(conversational_query, task_type="RETRIEVAL_QUERY")
            
            # --- 6.2. Retrieve top relevant chunks from the database ---
            top_k_chunks_to_retrieve = 3 # Changed back to 3 for more context, or keep at 2 if desired
            top_chunks_info = retrieve_top_k_chunks_from_chroma(query_embedding, top_k=top_k_chunks_to_retrieve)

            if not top_chunks_info:
                print("No relevant chunks found in the database.")
                retrieved_texts = []
                retrieved_image_paths_for_llm = []
                llm_context = "No specific context available from the document."
            else:
                print(f"\nTop {len(top_chunks_info)} retrieved chunks:")
                retrieved_texts = []
                retrieved_image_paths_for_llm = [] 

                for i, chunk_info in enumerate(top_chunks_info):
                    text = chunk_info['chunk_text']
                    image_paths = chunk_info['image_paths']
                    page_num = chunk_info['page_number']

                    retrieved_texts.append(text)
                    retrieved_image_paths_for_llm.extend(image_paths)
                    
                    print(f"--- Chunk {i+1} (Page {page_num+1}) ---")
                    print(text[:200] + "...")
                    if image_paths:
                        print(f"  [Associated Images: {', '.join([os.path.basename(p) for p in image_paths])}]")
                    print("-" * 30)

                retrieved_image_paths_for_llm = list(set(retrieved_image_paths_for_llm)) # Remove duplicates
                
                llm_context = "\n\n".join(retrieved_texts)

            # --- 6.3. Generate answer using LLM (Multimodal) ---
            print("\nGenerating answer with Gemini (multimodal)...")
            answer = await generate_answer_with_llm(query_text, llm_context, chat_history, retrieved_image_paths_for_llm)
            print("\n--- AI's Answer ---")
            print(answer)
            print("-------------------")

            # --- New Logic: Identify and Display Referenced Images ---
            referenced_images_in_answer = set() # Use a set to store unique paths
            # Regex to find patterns like 'Image: filename.ext (Page X)'
            # We are looking for something like "Image: page_X_img_Y.ext"
            image_ref_pattern = re.compile(r'Image: (page_\d+_img_\d+\.\w+)')

            # Iterate through the retrieved images that were sent to the LLM
            # and check if the LLM's answer contains a reference to their filename.
            for full_img_path in retrieved_image_paths_for_llm:
                filename = os.path.basename(full_img_path)
                if filename in answer: # Simple check: is filename directly in the answer text?
                    referenced_images_in_answer.add(full_img_path)
                
                # More robust check using regex
                # This ensures we catch "Image: filename (Page X)" patterns used by the LLM
                matches = image_ref_pattern.findall(answer)
                for matched_filename in matches:
                    if matched_filename == filename:
                        referenced_images_in_answer.add(full_img_path)

            if referenced_images_in_answer:
                print("\n--- AI Referenced These Images ---")
                for img_path in sorted(list(referenced_images_in_answer)): # Sort for consistent output
                    print(f"  - {img_path}")
                print("------------------------------------------")
            else:
                print("\n(The AI did not explicitly reference any specific images in its answer.)")

            # --- End New Logic ---

            # --- 6.4. Update Chat History ---
            chat_history.append((query_text, answer))
            chat_history = chat_history[-MAX_HISTORY_TURNS:]

            query_text = input("\nWrite another query (type 'exit' to quit): ")

    except Exception as e:
        print(f"An error occurred in main execution: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
        if images_dir and os.path.exists(images_dir):
            try:
                shutil.rmtree(images_dir)
                print(f"Cleaned up temporary image directory: {images_dir}")
            except Exception as e:
                print(f"Error cleaning up image directory {images_dir}: {e}")

if __name__ == "__main__":
    asyncio.run(main())

# C:\Users\Vedant\Downloads\1_12+Multimodal.pdf
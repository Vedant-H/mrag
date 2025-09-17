# import os
# import psycopg2
# from pgvector.psycopg2 import register_vector # Import the register_vector function
# import numpy as np
# from dotenv import load_dotenv
# import google.generativeai as genai
# from langchain_text_splitters import RecursiveCharacterTextSplitter # From your RAG setup
# import hashlib # For creating a simple document_id

# # --- Configuration ---
# load_dotenv()
# GOOGLE_API_KEY = "AIzaSyBEd1BvEtL_KWvGQaOFKIwMffZuZ9lVNtM"

# # PGVector Database Connection Details
# DB_HOST = "localhost"
# DB_PORT = "5432"
# DB_NAME = "postgres"
# DB_USER = "postgres"    # <--- IMPORTANT: Change this
# DB_PASSWORD = "Root@1234" #<-- IMPORTANT: Change this

# genai.configure(api_key=GOOGLE_API_KEY)

# # --- Helper Function to Get Embeddings ---
# def get_text_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
#     try:
#         response = genai.embed_content(
#             model="models/text-embedding-004",
#             content=text,
#             task_type=task_type # Can be RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY
#         )
#         return np.array(response['embedding'])
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         raise

# # --- PGVector Database Operations ---

# def get_db_connection():
#     """Establishes and returns a PostgreSQL database connection."""
#     conn = None
#     try:
#         conn = psycopg2.connect(
#             host=DB_HOST,
#             port=DB_PORT,
#             database=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD
#         )
#         # Register the vector type for psycopg2
#         register_vector(conn)
#         print("Successfully connected to the database.")
#         return conn
#     except psycopg2.Error as e:
#         print(f"Error connecting to database: {e}")
#         # In a real application, you might want to exit or log this more robustly
#         raise

# def create_table_if_not_exists(conn):
#     """Creates the document_chunks table if it doesn't already exist."""
#     with conn.cursor() as cur:
#         # Ensure the vector extension is available (run once per database)
#         cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#         conn.commit()

#         # Create the table with the 'vector' column
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS document_chunks (
#                 id SERIAL PRIMARY KEY,
#                 document_id VARCHAR(255) NOT NULL,
#                 chunk_text TEXT NOT NULL,
#                 embedding VECTOR(768) NOT NULL
#             );
#         """)
#         conn.commit()
#     print("Table 'document_chunks' ensured to exist.")

# def store_chunk_in_db(conn, document_id: str, chunk_text: str, embedding: np.ndarray):
#     """Stores a single text chunk and its embedding into the database."""
#     with conn.cursor() as cur:
#         # Convert numpy array to Python list for pgvector
#         embedding_list = embedding.tolist()
#         try:
#             cur.execute(
#                 "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES (%s, %s, %s)",
#                 (document_id, chunk_text, embedding_list)
#             )
#             conn.commit()
#             # print(f"Stored chunk for document '{document_id}'.")
#         except psycopg2.Error as e:
#             conn.rollback()
#             print(f"Error storing chunk: {e}")
#             raise

# def retrieve_top_k_chunks(conn, query_embedding: np.ndarray, top_k: int = 5):
#     """
#     Retrieves the top_k most similar document chunks based on a query embedding.
#     Uses cosine distance (equivalent to 1 - cosine similarity).
#     """
#     with conn.cursor() as cur:
#         query_embedding_list = query_embedding.tolist()
#         try:
#             # The '<=>' operator calculates L2 distance. For cosine similarity,
#             # you typically want 'cosine_distance(a, b)' which is '1 - cosine_similarity'.
#             # Or use '<=>' (L2) if you normalize vectors, as L2 then becomes equivalent to cosine.
#             # However, pgvector also directly supports cosine distance operator: '<=>' with normalized vectors
#             # or '1 - (a <#> b)' for cosine similarity. Let's use cosine_distance for clarity.

#             # IMPORTANT: For cosine similarity, pgvector recommends storing normalized vectors
#             # and then using L2 distance (<=>). Or use `1 - (a <#> b)` if not normalized.
#             # The text-embedding-004 model already produces normalized embeddings by default,
#             # so `<=>` (L2 distance) works well as a proxy for cosine distance.
#             # A smaller L2 distance means higher cosine similarity for normalized vectors.

#             # We will order by L2 distance ascending to get the most similar (smallest distance)
#             cur.execute(f"""
#                 SELECT id, document_id, chunk_text, embedding
#                 FROM document_chunks
#                 ORDER BY embedding <=> %s::vector
#                 LIMIT %s;
#             """, (query_embedding_list, top_k))
#             results = cur.fetchall()

#             # Process results: Convert embedding back to numpy array if needed
#             # For retrieval, we usually only care about the chunk_text
#             retrieved_chunks = []
#             for row in results:
#                 chunk_id, doc_id, text, embed = row
#                 retrieved_chunks.append({
#                     "id": chunk_id,
#                     "document_id": doc_id,
#                     "chunk_text": text,
#                     # "embedding": np.array(embed) # Uncomment if you need the retrieved embedding
#                 })
#             return retrieved_chunks
#         except psycopg2.Error as e:
#             print(f"Error retrieving chunks: {e}")
#             raise

# # --- Example Usage (How you'd use it to test and integrate) ---

# if __name__ == "__main__":
#     conn = None
#     try:
#         # 1. Connect to the database
#         conn = get_db_connection()

#         # 2. Ensure the table exists
#         create_table_if_not_exists(conn)

#         # 3. Simulate text chunking and embedding generation
#         sample_document_text = """
#         The quick brown fox jumps over the lazy dog. This is a very interesting sentence.
#         It contains several common English words. We will use this for testing our RAG system.
#         This is another paragraph about the importance of vector databases in modern AI applications.
#         They allow for efficient similarity search over large datasets of embeddings.
#         """
#         document_filename = "sample_document.pdf"
#         # Create a simple unique ID for the document (e.g., hash of filename + timestamp, or just filename)
#         doc_unique_id = hashlib.md5(document_filename.encode()).hexdigest()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=100, # Small chunks for this example
#             chunk_overlap=20,
#             length_function=len,
#         )
#         chunks = text_splitter.split_text(sample_document_text)

#         # print(f"\nProcessing {len(chunks)} chunks for document '{document_filename}'...")
#         # for i, chunk in enumerate(chunks):
#         #     print(f"  Chunk {i+1}: '{chunk[:50]}...'")
#         #     embedding = get_text_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
#         #     store_chunk_in_db(conn, doc_unique_id, chunk, embedding)
#         # print("All chunks processed and stored.")

#         # 4. Simulate a query and retrieve relevant chunks
#         query_text = input("write a query")
#         print(f"\nQuerying with: '{query_text}'")
#         query_embedding = get_text_embedding(query_text, task_type="RETRIEVAL_QUERY")
        
#         top_chunks = retrieve_top_k_chunks(conn, query_embedding, top_k=2)

#         print(f"\nTop {len(top_chunks)} retrieved chunks:")
#         for i, chunk_info in enumerate(top_chunks):
#             print(f"--- Chunk {i+1} (ID: {chunk_info['id']}, Doc ID: {chunk_info['document_id']}) ---")
#             print(chunk_info['chunk_text'])
#             print("-" * 30)

#     except Exception as e:
#         print(f"An error occurred in main execution: {e}")
#     finally:
#         if conn:
#             conn.close()
#             print("Database connection closed.")


import os
import sys
import fitz
import psycopg2
from pgvector.psycopg2 import register_vector # Import the register_vector function
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter # From your RAG setup
import hashlib # For creating a simple document_id

GOOGLE_API_KEY = "AIzaSyBEd1BvEtL_KWvGQaOFKIwMffZuZ9lVNtM"

# PGVector Database Connection Details
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"    # <--- IMPORTANT: Change this
DB_PASSWORD = "Root@1234" #<-- IMPORTANT: Change this

# genai.configure(api_key=GOOGLE_API_KEY)

# # --- Helper Function to Get Embeddings ---
# def get_text_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
#     """Generates an embedding for the given text using Google's text-embedding-004 model."""
#     try:
#         response = genai.embed_content(
#             model="models/text-embedding-004",
#             content=text,
#             task_type=task_type # Can be RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY
#         )
#         return np.array(response['embedding'])
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         raise

# # --- PGVector Database Operations ---

# def get_db_connection():
#     """Establishes and returns a PostgreSQL database connection."""
#     conn = None
#     try:
#         conn = psycopg2.connect(
#             host=DB_HOST,
#             port=DB_PORT,
#             database=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD
#         )
#         # Register the vector type for psycopg2
#         register_vector(conn)
#         print("Successfully connected to the database.")
#         return conn
#     except psycopg2.Error as e:
#         print(f"Error connecting to database: {e}")
#         # In a real application, you might want to exit or log this more robustly
#         raise

# def create_table_if_not_exists(conn):
#     """Creates the document_chunks table if it doesn't already exist."""
#     with conn.cursor() as cur:
#         # Ensure the vector extension is available (run once per database)
#         cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#         conn.commit()

#         # Create the table with the 'vector' column
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS document_chunks (
#                 id SERIAL PRIMARY KEY,
#                 document_id VARCHAR(255) NOT NULL,
#                 chunk_text TEXT NOT NULL,
#                 embedding VECTOR(768) NOT NULL
#             );
#         """)
#         conn.commit()
#     print("Table 'document_chunks' ensured to exist.")

# def clear_all_chunks_from_db(conn):
#     """Clears all data from the document_chunks table."""
#     with conn.cursor() as cur:
#         cur.execute("DELETE FROM document_chunks;")
#         conn.commit()
#     print("All existing chunks cleared from the database.")


# def store_chunk_in_db(conn, document_id: str, chunk_text: str, embedding: np.ndarray):
#     """Stores a single text chunk and its embedding into the database."""
#     with conn.cursor() as cur:
#         # Convert numpy array to Python list for pgvector
#         embedding_list = embedding.tolist()
#         try:
#             cur.execute(
#                 "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES (%s, %s, %s)",
#                 (document_id, chunk_text, embedding_list)
#             )
#             conn.commit()
#             # print(f"Stored chunk for document '{document_id}'.") # Commented for less verbosity
#         except psycopg2.Error as e:
#             conn.rollback()
#             print(f"Error storing chunk: {e}")
#             raise

# def retrieve_top_k_chunks(conn, query_embedding: np.ndarray, top_k: int = 5):
#     """
#     Retrieves the top_k most similar document chunks based on a query embedding.
#     Uses L2 distance (<=>) as cosine similarity proxy for normalized vectors.
#     """
#     with conn.cursor() as cur:
#         query_embedding_list = query_embedding.tolist()
#         try:
#             cur.execute(f"""
#                 SELECT id, document_id, chunk_text, embedding
#                 FROM document_chunks
#                 ORDER BY embedding <=> %s::vector
#                 LIMIT %s;
#             """, (query_embedding_list, top_k))
#             results = cur.fetchall()

#             retrieved_chunks = []
#             for row in results:
#                 chunk_id, doc_id, text, embed = row
#                 retrieved_chunks.append({
#                     "id": chunk_id,
#                     "document_id": doc_id,
#                     "chunk_text": text,
#                     # "embedding": np.array(embed) # Uncomment if you need the retrieved embedding
#                 })
#             return retrieved_chunks
#         except psycopg2.Error as e:
#             print(f"Error retrieving chunks: {e}")
#             raise

# # --- LLM Generation Function ---
# async def generate_answer_with_llm(question: str, retrieved_context: str) -> str:
#     """
#     Generates an answer using Google Gemini 1.5 Flash based on the question and retrieved context.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         prompt = f"""
#         You are a helpful and precise assistant for question-answering based on provided context.
#         Use ONLY the following context to answer the question. If the answer cannot be found in the context,
#         state that you cannot answer based on the provided information. Do not make up answers.

#         Context:
#         {retrieved_context}

#         Question:
#         {question}

#         Answer:
#         """
#         response = await model.generate_content_async(prompt)
#         return response.text
#     except Exception as e:
#         print(f"Error generating answer with LLM: {e}")
#         return "Sorry, I couldn't generate an answer due to an internal error."


# # --- PDF Text Extraction Function ---
# def load_pdf_text(pdf_path: str) -> str:
#     """Extracts text from a PDF file."""
#     text = ""
#     try:
#         document = fitz.open(pdf_path)
#         for page_num in range(len(document)):
#             page = document.load_page(page_num)
#             text += page.get_text()
#         print(f"Successfully extracted text from: {pdf_path}")
#         return text
#     except fitz.FileDataError:
#         print(f"Error: Could not open or read PDF file at {pdf_path}. It might be corrupted or not a valid PDF.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"An unexpected error occurred while reading PDF: {e}")
#         sys.exit(1)


# # --- Main Execution Block (for testing) ---
# async def main():
#     conn = None
#     try:
#         # 1. Connect to the database
#         conn = get_db_connection()

#         # 2. Ensure the table exists
#         # create_table_if_not_exists(conn)

#         # 3. Clear existing data (optional, for a clean start each time)
#         # clear_all_chunks_from_db(conn)
#         document_text = load_pdf_text(r'C:\Users\Vedant\Downloads\test.pdf')
#         # 4. Simulate text chunking and embedding generation for a document
#         sample_document_text = document_text
#         document_filename = "fables_and_vectors_intro.txt"
#         doc_unique_id = hashlib.md5(document_filename.encode()).hexdigest()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500, # Increased size for better semantic coherence in research papers
#             chunk_overlap=300, # Sufficient overlap to maintain context across chunks
#             length_function=len, # Still counting characters
#         )
#         chunks = text_splitter.split_text(sample_document_text)

#         print(f"\nProcessing {len(chunks)} chunks for document '{document_filename}'...")
#         for i, chunk in enumerate(chunks):
#             embedding = get_text_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
#             store_chunk_in_db(conn, doc_unique_id, chunk, embedding)
#         print("All chunks processed and stored in the database.")

#         # 5. User provides a query
#         query_text = input("\nWrite a query about the document (type 'exit' to quit): ")

#         while query_text.lower() != 'exit':
#             print(f"\nQuerying with: '{query_text}'")
#             query_embedding = get_text_embedding(query_text, task_type="RETRIEVAL_QUERY")
            
#             # Retrieve top relevant chunks from the database
#             top_chunks_info = retrieve_top_k_chunks(conn, query_embedding, top_k=5)

#             if not top_chunks_info:
#                 print("No relevant chunks found in the database.")
#                 llm_context = "No specific context available from the document."
#             else:
#                 # print(f"\nTop {len(top_chunks_info)} retrieved chunks:")
#                 retrieved_texts = [chunk_info['chunk_text'] for chunk_info in top_chunks_info]
#                 llm_context = "\n\n".join(retrieved_texts)

#             # Generate answer using LLM
#             print("\nGenerating answer with Gemini...")
#             answer = await generate_answer_with_llm(query_text, llm_context)
#             print("\n--- AI's Answer ---")
#             print(answer)
#             print("-------------------")

#             query_text = input("\nWrite another query (type 'exit' to quit): ")

#     except Exception as e:
#         print(f"An error occurred in main execution: {e}")
#     finally:
#         if conn:
#             conn.close()
#             print("Database connection closed.")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper Function to Get Embeddings ---
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

# --- PDF Text Extraction Function ---
def load_pdf_text(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        print(f"Successfully extracted text from: {pdf_path}")
        return text
    except fitz.FileDataError:
        print(f"Error: Could not open or read PDF file at {pdf_path}. It might be corrupted or not a valid PDF.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading PDF: {e}")
        sys.exit(1)

# --- PGVector Database Operations ---

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        register_vector(conn)
        print("Successfully connected to the database.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise

def create_table_if_not_exists(conn):
    """
    Creates the document_chunks table if it doesn't already exist,
    and also creates an HNSW index on the embedding column.
    """
    with conn.cursor() as cur:
        # Ensure the vector extension is available (run once per database)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        # Create the table with the 'vector' column
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id VARCHAR(255) NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding VECTOR(768) NOT NULL
            );
        """)
        conn.commit()
        print("Table 'document_chunks' ensured to exist.")

        # --- Add HNSW Index ---
        # This creates an HNSW index on the 'embedding' column for cosine distance.
        # It's crucial for performance on large datasets.
        # The 'IF NOT EXISTS' ensures it won't throw an error if the index already exists.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """)
        conn.commit()
        print("HNSW index 'idx_document_chunks_embedding' ensured to exist.")


def clear_all_chunks_from_db(conn):
    """Clears all data from the document_chunks table."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks;")
        conn.commit()
    print("All existing chunks cleared from the database.")


def store_chunk_in_db(conn, document_id: str, chunk_text: str, embedding: np.ndarray):
    """Stores a single text chunk and its embedding into the database."""
    with conn.cursor() as cur:
        embedding_list = embedding.tolist()
        try:
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES (%s, %s, %s)",
                (document_id, chunk_text, embedding_list)
            )
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            print(f"Error storing chunk: {e}")
            raise

def retrieve_top_k_chunks(conn, query_embedding: np.ndarray, top_k: int = 5):
    """
    Retrieves the top_k most similar document chunks based on a query embedding.
    Uses L2 distance (<=>) as cosine similarity proxy for normalized vectors.
    """
    with conn.cursor() as cur:
        query_embedding_list = query_embedding.tolist()
        try:
            cur.execute(f"""
                SELECT id, document_id, chunk_text, embedding
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding_list, top_k))
            results = cur.fetchall()

            retrieved_chunks = []
            for row in results:
                chunk_id, doc_id, text, embed = row
                retrieved_chunks.append({
                    "id": chunk_id,
                    "document_id": doc_id,
                    "chunk_text": text,
                })
            return retrieved_chunks
        except psycopg2.Error as e:
            print(f"Error retrieving chunks: {e}")
            raise

# --- LLM Generation Function ---
# async def generate_answer_with_llm(question: str, retrieved_context: str) -> str:
#     """
#     Generates an answer using Google Gemini 1.5 Flash based on the question and retrieved context.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         prompt = f"""
#         You are a helpful and precise assistant for question-answering based on provided context.
#         Use ONLY the following context to answer the question. If the answer cannot be found in the context,
#         state that you cannot answer based on the provided information. Do not make up answers.

#         Context:
#         {retrieved_context}

#         Question:
#         {question}

#         Answer:
#         """
#         response = await model.generate_content_async(prompt)
#         return response.text
#     except Exception as e:
#         print(f"Error generating answer with LLM: {e}")
#         return "Sorry, I couldn't generate an answer due to an internal error."
async def generate_answer_with_llm(question: str, retrieved_context: str, chat_history: list) -> str:
    """
    Generates an answer using Google Gemini 1.5 Flash based on the question, retrieved context,
    and previous chat history.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Format chat history for the prompt
        history_str = ""
        if chat_history:
            history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
            history_str = "--- Chat History ---\n" + history_str + "\n--------------------"

        prompt = f"""
        You are a helpful and precise assistant for question-answering based on provided context and chat history.
        Use ONLY the following context and the chat history to answer the question.
        If the answer cannot be found in the provided context, you may use your general knowledge
        to provide a concise and relevant answer based on the context. Do not make up answers.

        {history_str}

        Context:
        {retrieved_context}

        Question:
        {question}

        Answer:
        """
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        return "Sorry, I couldn't generate an answer due to an internal error."

# --- Main Execution Block (for testing) ---

async def main():
    conn = None
    try:
        conn = get_db_connection()
        create_table_if_not_exists(conn)

        # pdf_path = input("Enter the path to your PDF file (e.g., 'document.pdf'): ")
        # if not os.path.exists(pdf_path):
        #     print(f"Error: File not found at {pdf_path}")
        #     sys.exit(1)
        # if not pdf_path.lower().endswith(".pdf"):
        #     print(f"Error: The provided file is not a PDF. Please provide a .pdf file.")
        #     sys.exit(1)

        # clear_all_chunks_from_db(conn)

        # document_text = load_pdf_text(pdf_path)
        # if not document_text.strip():
        #     print("Error: Extracted text from PDF is empty. Cannot process.")
        #     sys.exit(1)

        # doc_unique_id = hashlib.md5(document_text.encode('utf-8')).hexdigest()
        # document_filename = os.path.basename(pdf_path)

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1500,
        #     chunk_overlap=300,
        #     length_function=len,
        # )
        # chunks = text_splitter.split_text(document_text)

        # print(f"\nProcessing {len(chunks)} chunks for document '{document_filename}' (ID: {doc_unique_id[:8]}...)...")
        # for i, chunk in enumerate(chunks):
        #     embedding = get_text_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
        #     store_chunk_in_db(conn, doc_unique_id, chunk, embedding)
        # print("All chunks processed and stored in the database.")

        # --- Initialize Chat History ---
        chat_history = []
        MAX_HISTORY_TURNS = 3 # Store previous 3 user-AI turns

        query_text = input("\nWrite a query about the document (type 'exit' to quit): ")

        while query_text.lower() != 'exit':
            # --- 1. Incorporate history for Query Embedding ---
            # Create a conversational query by combining recent history and current question
            conversational_query_parts = []
            # Add recent history to the query for better embedding, keeping it concise
            for q, a in chat_history[-MAX_HISTORY_TURNS:]: # Only use the last few turns
                conversational_query_parts.append(f"Q: {q} A: {a}")
            conversational_query_parts.append(f"Current Q: {query_text}")

            conversational_query = " ".join(conversational_query_parts)
            print(f"\nEmbedding query with context: '{conversational_query}'")

            query_embedding = get_text_embedding(conversational_query, task_type="RETRIEVAL_QUERY")

            top_chunks_info = retrieve_top_k_chunks(conn, query_embedding, top_k=3)

            if not top_chunks_info:
                print("No relevant chunks found in the database.")
                retrieved_texts = []
                llm_context = "No specific context available from the document."
            else:
                print(f"\nTop {len(top_chunks_info)} retrieved chunks:")
                retrieved_texts = [chunk_info['chunk_text'] for chunk_info in top_chunks_info]
                for i, text in enumerate(retrieved_texts):
                    print(f"--- Chunk {i+1} ---")
                    print(text[:200] + "...")
                    print("-" * 30)
                llm_context = "\n\n".join(retrieved_texts)

            # --- 2. Generate Answer with LLM, passing full chat history ---
            print("\nGenerating answer with Gemini...")
            answer = await generate_answer_with_llm(query_text, llm_context, chat_history) # Pass chat_history
            print("\n--- AI's Answer ---")
            print(answer)
            print("-------------------")

            # --- Update Chat History ---
            chat_history.append((query_text, answer))
            # Keep only the most recent 'MAX_HISTORY_TURNS' in memory
            chat_history = chat_history[-MAX_HISTORY_TURNS:]

            query_text = input("\nWrite another query (type 'exit' to quit): ")

    except Exception as e:
        print(f"An error occurred in main execution: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
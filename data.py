# import json
# import os
# import numpy as np
# import psycopg2
# from pgvector.psycopg2 import register_vector 
# from dotenv import load_dotenv 

# load_dotenv() 

# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "5432")
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")


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
#     """
#     Creates the document_chunks1 table if it doesn't already exist,
#     and also creates an HNSW index on the embedding column.
#     """
#     with conn.cursor() as cur:
#         # Ensure the vector extension is available (run once per database)
#         cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#         conn.commit()

#         # Create the table with the 'vector' column and new columns for multimodal support
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS document_chunks1 (
#                 id SERIAL PRIMARY KEY,
#                 document_id VARCHAR(255) NOT NULL,
#                 chunk_text TEXT NOT NULL,
#                 embedding VECTOR(768) NOT NULL,
#                 image_paths_json TEXT, -- New column to store JSON array of image file paths
#                 page_number INTEGER -- New column to store original page number for context
#             );
#         """)
#         conn.commit()
#         print("Table 'document_chunks1' ensured to exist.")

#         # Create HNSW index on the 'embedding' column for cosine distance
#         cur.execute("""
#             CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks1 USING hnsw (embedding vector_cosine_ops);
#         """)
#         conn.commit()
#         print("HNSW index 'idx_document_chunks_embedding' ensured to exist.")


# def clear_all_chunks_from_db(conn):
#     """Clears all data from the document_chunks1 table."""
#     with conn.cursor() as cur:
#         cur.execute("DELETE FROM document_chunks1;")
#         conn.commit()
#     print("All existing chunks cleared from the database.")


# def store_chunk_in_db(conn, document_id: str, chunk_text: str, embedding: np.ndarray, image_paths: list = None, page_number: int = None):
#     """Stores a single text chunk, its embedding, associated image paths, and page number into the database."""
#     embedding_list = embedding.tolist()
#     # Convert list of image paths to a JSON string for storage
#     image_paths_json_str = json.dumps(image_paths) if image_paths else None
    
#     with conn.cursor() as cur:
#         try:
#             cur.execute(
#                 "INSERT INTO document_chunks1 (document_id, chunk_text, embedding, image_paths_json, page_number) VALUES (%s, %s, %s, %s, %s)",
#                 (document_id, chunk_text, embedding_list, image_paths_json_str, page_number)
#             )
#             conn.commit()
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
#                 SELECT id, document_id, chunk_text, embedding, image_paths_json, page_number
#                 FROM document_chunks1
#                 ORDER BY embedding <=> %s::vector
#                 LIMIT %s;
#             """, (query_embedding_list, top_k))
#             results = cur.fetchall()

#             retrieved_chunks = []
#             for row in results:
#                 chunk_id, doc_id, text, embed, image_paths_json_str, page_number = row
                
#                 # Parse image paths back from JSON string
#                 image_paths = json.loads(image_paths_json_str) if image_paths_json_str else []

#                 retrieved_chunks.append({
#                     "id": chunk_id,
#                     "document_id": doc_id,
#                     "chunk_text": text,
#                     "image_paths": image_paths, # Store the list of paths
#                     "page_number": page_number
#                 })
#             return retrieved_chunks
#         except psycopg2.Error as e:
#             print(f"Error retrieving chunks: {e}")
#             raise

# data.py (Updated content)
import json
import os
import numpy as np
import chromadb
import hashlib # You'll need this for ID generation
from dotenv import load_dotenv

load_dotenv()

# We will use the PersistentClient to save data to a local directory
CHROMA_DB_PATH = "chroma_db_data"

def get_db_collection():
    """Initializes and returns a ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name="document_chunks1")
        print("Successfully connected to ChromaDB and collection 'document_chunks1' is ready.")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        raise

# data.py
def clear_all_chunks_from_db(collection):
    """Clears all data from the specified collection."""
    try:
        # Pass an empty filter to delete all documents.
        collection.delete(where={})
        print("All existing chunks cleared from the database.")
    except Exception as e:
        print(f"Error clearing collection: {e}")



def store_chunk_in_db(collection, document_id: str, chunk_text: str, embedding: np.ndarray, image_paths: list = None, page_number: int = None):
    """Stores a single text chunk, its embedding, associated image paths, and page number into the database."""
    # ... (your existing code to generate chunk_id) ...
    chunk_id = f"{document_id}_{page_number}_{hashlib.sha1(chunk_text.encode()).hexdigest()[:8]}"

    # ChromaDB stores metadata as a dictionary
    metadata = {
        "document_id": document_id,
        "chunk_text": chunk_text,
        # Ensure image_paths is always a list, even if None is passed.
        # Then, if the list is empty, dump it as '[]' rather than None.
        "image_paths_json": json.dumps(image_paths) if image_paths is not None else "[]",
        "page_number": page_number,
    }

    try:
        # Use the .add() method to store the vector and metadata
        collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[chunk_text],
            ids=[chunk_id]
        )
    except Exception as e:
        print(f"Error storing chunk: {e}")
        raise

def retrieve_top_k_chunks(collection, query_embedding: np.ndarray, top_k: int = 5):
    """Retrieves the top_k most similar document chunks based on a query embedding."""
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['metadatas'] # We only need the metadata, not the embeddings or distances
        )

        retrieved_chunks = []
        for metadata in results['metadatas'][0]:
            image_paths_json_str = metadata.get("image_paths_json")
            image_paths = json.loads(image_paths_json_str) if image_paths_json_str else []
            
            retrieved_chunks.append({
                "document_id": metadata.get("document_id"),
                "chunk_text": metadata.get("chunk_text"),
                "image_paths": image_paths,
                "page_number": metadata.get("page_number")
            })
        return retrieved_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        raise
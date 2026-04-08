import os
import re
import glob
from typing import List
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_genai_client():
    """
    Returns (sdk_type, client) tuple.
    Supports both the new `google-genai` and old `google-generativeai` SDKs.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Try new SDK first  (pip install google-genai)
    try:
        from google import genai as new_genai
        client = new_genai.Client(api_key=api_key)
        return ("new", client)
    except (ImportError, AttributeError):
        pass

    # Fall back to old SDK  (pip install google-generativeai)
    try:
        import google.generativeai as old_genai
        old_genai.configure(api_key=api_key)
        return ("old", old_genai)
    except ImportError:
        raise ImportError(
            "Neither `google-genai` nor `google-generativeai` is installed.\n"
            "Run:  pip install google-genai"
        )


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using gemini-embedding-001 with output_dimensionality=768
    so it matches an existing ChromaDB collection built with 768-dim embeddings.
    """
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    OUTPUT_DIM = 768  # ← Option 3: force 768 dims to match existing ChromaDB collection

    def __call__(self, input: Documents) -> Embeddings:
        sdk_type, client = _get_genai_client()

        try:
            if sdk_type == "new":
                # NEW google-genai SDK
                embeddings = []
                for text in input:
                    response = client.models.embed_content(
                        model=self.EMBEDDING_MODEL,
                        contents=text,
                        config={
                            "task_type": "retrieval_document",
                            "title": "Document vectorization",
                            "output_dimensionality": self.OUTPUT_DIM,  # ← key fix
                        },
                    )
                    embeddings.append(response.embeddings[0].values)
                return embeddings

            else:
                # OLD google-generativeai SDK
                response = client.embed_content(
                    model=self.EMBEDDING_MODEL,
                    content=input,
                    task_type="retrieval_document",
                    title="Document vectorization",
                    output_dimensionality=self.OUTPUT_DIM,  # ← key fix
                )
                return response["embedding"]

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class PDFVectorizer:
    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_function = GeminiEmbeddingFunction()

    def load_pdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.info(f"Successfully loaded PDF: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def process_pdf_folder(self, pdf_folder_path: str) -> List[dict]:
        pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        all_documents = []

        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file}")
            pdf_text = self.load_pdf(pdf_file)
            if not pdf_text.strip():
                logger.warning(f"No text extracted from {pdf_file}")
                continue

            chunks = self.split_text(pdf_text)
            logger.info(f"Created {len(chunks)} chunks from {os.path.basename(pdf_file)}")

            for i, chunk in enumerate(chunks):
                all_documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': os.path.basename(pdf_file),
                        'full_path': pdf_file,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                })

        logger.info(f"Total documents created: {len(all_documents)}")
        return all_documents

    def create_vector_database(self, documents: List[dict]) -> chromadb.Collection:
        try:
            chroma_client = chromadb.PersistentClient(path=self.db_path)

            try:
                chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass

            db = chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_documents = [doc['content'] for doc in batch]
                batch_ids = [f"doc_{i + j}" for j in range(len(batch))]
                batch_metadatas = [doc['metadata'] for doc in batch]

                logger.info(f"Adding batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

                try:
                    db.add(
                        documents=batch_documents,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                except Exception as e:
                    logger.error(f"Error adding batch to database: {e}")
                    continue

            logger.info(f"Successfully created vector database with {len(documents)} documents")
            return db

        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise

    def load_existing_database(self) -> chromadb.Collection:
        try:
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            db = chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Successfully loaded existing database: {self.collection_name}")
            return db
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            raise

    def query_database(self, db: chromadb.Collection, query: str, n_results: int = 5) -> dict:
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                results = db.query(
                    query_texts=[query],
                    n_results=n_results
                )
                return results
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Too Many Demand" in err_str or "quota" in err_str.lower() or "Rate" in err_str:
                    if attempt < max_retries - 1:
                        wait_t = (2 ** attempt) * 5
                        logger.warning(f"Gemini API rate limit hit during query. Retrying in {wait_t}s...")
                        time.sleep(wait_t)
                        continue
                logger.error(f"Error querying database: {e}")
                raise


def main():
    PDF_FOLDER_PATH = r"D:\CAPSTONE\RAG-resource"
    DB_PATH = r"./vector_db"
    COLLECTION_NAME = r"pdf_rag_collection"

    if not os.getenv("GEMINI_API_KEY"):
        logger.error("Please set GEMINI_API_KEY environment variable")
        return

    try:
        vectorizer = PDFVectorizer(db_path=DB_PATH, collection_name=COLLECTION_NAME)

        logger.info("Starting PDF processing...")
        documents = vectorizer.process_pdf_folder(PDF_FOLDER_PATH)

        if not documents:
            logger.error("No documents to process")
            return

        logger.info("Creating vector database...")
        db = vectorizer.create_vector_database(documents)

        logger.info("Testing database with sample query...")
        test_query = "What is the main topic discussed in the documents?"
        results = vectorizer.query_database(db, test_query, n_results=3)

        print("\n=== SAMPLE QUERY RESULTS ===")
        print(f"Query: {test_query}")
        print("\nTop 3 Results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\n{i + 1}. Source: {metadata['source']}")
            print(f"   Chunk {metadata['chunk_id']}/{metadata['total_chunks']}")
            print(f"   Content: {doc[:200]}...")

        logger.info("Vector database creation completed successfully!")

    except Exception as e:
        logger.error(f"Error in main process: {e}")


if __name__ == "__main__":
    main()
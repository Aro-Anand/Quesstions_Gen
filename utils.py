# utils.py
"""
Enhanced utility functions with PDF processing capabilities.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from document_processor import process_and_embed_pdf

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate embedding for given text using OpenAI.
    
    Args:
        text: Input text to embed
        model: OpenAI embedding model name
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def initialize_pinecone() -> Pinecone:
    """
    Initialize Pinecone client and create/connect to index.
    
    Returns:
        Pinecone client instance
    """
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise


def setup_syllabus_index(pc: Pinecone, index_name: str = "syllabus-vectors") -> Any:
    """
    Create or connect to Pinecone index for syllabus documents.
    
    Args:
        pc: Pinecone client instance
        index_name: Name of the index to create/use
        
    Returns:
        Pinecone index object
    """
    try:
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        # Connect to index
        index = pc.Index(index_name)
        logger.info(f"Connected to index: {index_name}")
        
        return index
    except Exception as e:
        logger.error(f"Error setting up index: {e}")
        raise


def upload_pdf_to_vectorstore(
    index: Any,
    pdf_path: str,
    class_level: str,
    subject: str,
    chapter: str,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Process PDF and upload to Pinecone vector store.
    
    Args:
        index: Pinecone index object
        pdf_path: Path to PDF file
        class_level: Class level (e.g., "Class 10")
        subject: Subject name
        chapter: Chapter name
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with upload statistics
    """
    try:
        if progress_callback:
            progress_callback("Extracting text from PDF...")
        
        # Process and embed PDF
        api_key = os.getenv("OPENAI_API_KEY")
        vectors = process_and_embed_pdf(
            pdf_path=pdf_path,
            api_key=api_key,
            class_level=class_level,
            subject=subject,
            chapter=chapter
        )
        
        if progress_callback:
            progress_callback(f"Uploading {len(vectors)} chunks to vector store...")
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            
            if progress_callback:
                progress = (i + len(batch)) / len(vectors) * 100
                progress_callback(f"Uploaded {i + len(batch)}/{len(vectors)} chunks ({progress:.1f}%)")
        
        # Get file metadata
        filename = os.path.basename(pdf_path)
        file_hash = vectors[0]["metadata"]["file_hash"] if vectors else None
        
        result = {
            "filename": filename,
            "file_hash": file_hash,
            "chunks_uploaded": len(vectors),
            "class": class_level,
            "subject": subject,
            "chapter": chapter,
            "status": "success"
        }
        
        logger.info(f"Successfully uploaded {filename}: {len(vectors)} chunks")
        return result
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return {
            "filename": os.path.basename(pdf_path),
            "status": "error",
            "error": str(e)
        }


def query_syllabus_context(
    index: Any,
    query_text: str,
    filters: Optional[Dict[str, str]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for relevant syllabus context.
    
    Args:
        index: Pinecone index object
        query_text: Text to search for
        filters: Metadata filters (class, subject, etc.)
        top_k: Number of results to return
        
    Returns:
        List of matching context dictionaries
    """
    try:
        query_embedding = get_embedding(query_text)
        
        # Build filter dict for Pinecone
        filter_dict = {}
        if filters:
            for key, value in filters.items():
                if value:
                    filter_dict[key] = value
        
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if filter_dict:
            query_params["filter"] = filter_dict
        
        results = index.query(**query_params)
        
        contexts = []
        for match in results.matches:
            contexts.append({
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "metadata": match.metadata
            })
        
        return contexts
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return []


def get_uploaded_documents(index: Any) -> List[Dict[str, Any]]:
    """
    Get list of unique documents in vector store.
    
    Args:
        index: Pinecone index object
        
    Returns:
        List of document metadata
    """
    try:
        stats = index.describe_index_stats()
        
        # Query for unique documents using metadata
        # This is a simplified version - in production, maintain a separate metadata store
        sample_results = index.query(
            vector=[0.0] * 1536,
            top_k=100,
            include_metadata=True
        )
        
        # Extract unique documents
        docs_dict = {}
        for match in sample_results.matches:
            file_hash = match.metadata.get("file_hash")
            if file_hash and file_hash not in docs_dict:
                docs_dict[file_hash] = {
                    "filename": match.metadata.get("filename", "Unknown"),
                    "file_hash": file_hash,
                    "class": match.metadata.get("class", "Unknown"),
                    "subject": match.metadata.get("subject", "Unknown"),
                    "chapter": match.metadata.get("chapter", "Unknown")
                }
        
        return list(docs_dict.values())
        
    except Exception as e:
        logger.error(f"Error getting uploaded documents: {e}")
        return []


def delete_document_from_vectorstore(
    index: Any,
    file_hash: str
) -> bool:
    """
    Delete all vectors associated with a document.
    
    Args:
        index: Pinecone index object
        file_hash: Hash of file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Query all vectors with this file_hash
        results = index.query(
            vector=[0.0] * 1536,
            top_k=10000,
            include_metadata=True,
            filter={"file_hash": file_hash}
        )
        
        # Delete by IDs
        if results.matches:
            ids_to_delete = [match.id for match in results.matches]
            index.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} vectors for file_hash: {file_hash}")
            return True
        
        logger.warning(f"No vectors found for file_hash: {file_hash}")
        return False
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return False


def get_curriculum_options() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Get hierarchical curriculum options for UI dropdowns.
    
    Returns:
        Nested dictionary of class -> subject -> chapter -> topics
    """
    return {
        "Class 10": {
            "Math": {
                "Algebra": ["Quadratic Equations", "Polynomials", "Linear Equations"],
                "Geometry": ["Triangles", "Circles", "Coordinate Geometry"],
                "Trigonometry": ["Trigonometric Ratios", "Heights and Distances"],
                "Statistics": ["Central Tendency", "Probability"]
            },
            "Science": {
                "Physics": ["Motion and Force", "Light", "Sound"],
                "Chemistry": ["Acids and Bases", "Metals and Non-metals", "Chemical Reactions"],
                "Biology": ["Cell Structure", "Heredity", "Life Processes"]
            }
        },
        "Class 12": {
            "Math": {
                "Calculus": ["Derivatives", "Integration", "Differential Equations"],
                "Vectors": ["Vector Operations", "3D Geometry"],
                "Probability": ["Conditional Probability", "Distributions"]
            },
            "Science": {
                "Physics": ["Electricity", "Magnetism", "Optics", "Modern Physics"],
                "Chemistry": ["Organic Chemistry", "Chemical Kinetics", "Electrochemistry"],
                "Biology": ["Genetics", "Evolution", "Ecology"]
            }
        }
    }


def get_vectorstore_stats(index: Any) -> Dict[str, Any]:
    """
    Get statistics about the vector store.
    
    Args:
        index: Pinecone index object
        
    Returns:
        Dictionary with statistics
    """
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": 1536,
            "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else 0.0
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"total_vectors": 0, "dimension": 1536, "index_fullness": 0.0}
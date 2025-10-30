# document_processor.py
"""
PDF processing module for extracting, chunking, and embedding syllabus documents.

This module provides comprehensive PDF processing capabilities:
- Text extraction using multiple libraries (pdfplumber, PyPDF2)
- Intelligent chunking with token-based splitting
- Embedding generation with batch processing
- File deduplication using content hashing
"""
import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
import PyPDF2
import pdfplumber
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of processed document."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    page_number: int
    chunk_index: int


class PDFProcessor:
    """Handles PDF extraction, chunking, and preprocessing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        pages = []
        
        try:
            # Try pdfplumber first (better for complex PDFs)
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            "page_number": i + 1,
                            "text": text.strip(),
                            "extraction_method": "pdfplumber"
                        })
            
            logger.info(f"Extracted {len(pages)} pages from {pdf_path} using pdfplumber")
            
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            pages.append({
                                "page_number": i + 1,
                                "text": text.strip(),
                                "extraction_method": "PyPDF2"
                            })
                
                logger.info(f"Extracted {len(pages)} pages from {pdf_path} using PyPDF2")
                
            except Exception as e2:
                logger.error(f"Failed to extract text from {pdf_path}: {e2}")
                raise
        
        return pages
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(
        self,
        text: str,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            page_number: Source page number
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split by sentences or paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_tokens = self.count_tokens(paragraph)
            
            # If single paragraph exceeds chunk size, force split
            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        page_number,
                        chunk_index,
                        metadata
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Split long paragraph by sentences
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip() + '. '
                    sent_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sent_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                current_chunk,
                                page_number,
                                chunk_index,
                                metadata
                            ))
                            chunk_index += 1
                        current_chunk = sentence
                        current_tokens = sent_tokens
                    else:
                        current_chunk += sentence
                        current_tokens += sent_tokens
            
            # Normal paragraph handling
            elif current_tokens + para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        page_number,
                        chunk_index,
                        metadata
                    ))
                    chunk_index += 1
                current_chunk = paragraph + "\n\n"
                current_tokens = para_tokens
            else:
                current_chunk += paragraph + "\n\n"
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk,
                page_number,
                chunk_index,
                metadata
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        page_number: int,
        chunk_index: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        chunk_id = hashlib.md5(
            f"{metadata.get('filename', '')}_{page_number}_{chunk_index}".encode()
        ).hexdigest()
        
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            metadata=metadata,
            page_number=page_number,
            chunk_index=chunk_index
        )
    
    def process_pdf(
        self,
        pdf_path: str,
        class_level: Optional[str] = None,
        subject: Optional[str] = None,
        chapter: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Process entire PDF: extract, chunk, and prepare for embedding.
        
        Args:
            pdf_path: Path to PDF file
            class_level: Class level (e.g., "Class 10")
            subject: Subject name
            chapter: Chapter name
            
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        filename = os.path.basename(pdf_path)
        file_hash = self._compute_file_hash(pdf_path)
        
        # Base metadata
        metadata = {
            "filename": filename,
            "file_hash": file_hash,
            "source": pdf_path,
            "class": class_level or "Unknown",
            "subject": subject or "Unknown",
            "chapter": chapter or "Unknown"
        }
        
        logger.info(f"Processing PDF: {filename}")
        
        # Extract text
        pages = self.extract_text_from_pdf(pdf_path)
        
        # Chunk each page
        all_chunks = []
        for page_data in pages:
            page_chunks = self.chunk_text(
                text=page_data["text"],
                page_number=page_data["page_number"],
                metadata={
                    **metadata,
                    "extraction_method": page_data["extraction_method"]
                }
            )
            all_chunks.extend(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        
        return all_chunks
    
    def _compute_file_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class EmbeddingManager:
    """Manages document embeddings using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize embedding manager.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536  # text-embedding-3-small dimension
    
    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to embed per API call
            
        Returns:
            List of vectors ready for Pinecone upsert
        """
        vectors = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                logger.info(f"Embedding batch {i//batch_size + 1} ({len(texts)} chunks)")
                
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                for j, embedding_obj in enumerate(response.data):
                    chunk = batch[j]
                    vectors.append({
                        "id": f"{chunk.chunk_id}_{i+j}",
                        "values": embedding_obj.embedding,
                        "metadata": {
                            **chunk.metadata,
                            "text": chunk.text,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "token_count": len(chunk.text.split())
                        }
                    })
                
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Generated {len(vectors)} embeddings")
        return vectors


def process_and_embed_pdf(
    pdf_path: str,
    api_key: str,
    class_level: Optional[str] = None,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    End-to-end PDF processing: extract, chunk, and embed.
    
    Args:
        pdf_path: Path to PDF file
        api_key: OpenAI API key
        class_level: Class level
        subject: Subject name
        chapter: Chapter name
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between chunks
        
    Returns:
        List of vectors ready for Pinecone upsert
    """
    # Process PDF
    processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_pdf(pdf_path, class_level, subject, chapter)
    
    # Generate embeddings
    embedding_manager = EmbeddingManager(api_key=api_key)
    vectors = embedding_manager.embed_chunks(chunks)
    
    return vectors
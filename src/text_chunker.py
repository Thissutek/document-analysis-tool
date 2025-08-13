"""
Text Chunker Module
Handles document segmentation with token-based chunking
"""
from typing import List, Dict, Optional
import streamlit as st
import tiktoken
import re


class TextChunker:
    """Segments documents into manageable chunks based on token count for AI processing"""
    
    def __init__(self, chunk_tokens: int = 1000, overlap_tokens: int = 100):
        """
        Initialize text chunker with token-based sizing
        
        Args:
            chunk_tokens: Number of tokens per chunk (default 1000 for AI processing)
            overlap_tokens: Number of tokens to overlap between chunks
        """
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize tiktoken encoder for GPT models
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")  # Used by GPT-3.5/GPT-4
        except Exception as e:
            st.error(f"Error initializing token encoder: {e}")
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        if not self.encoder or not text:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
        
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # Fallback calculation
            return len(text) // 4
    
    def chunk_text_by_tokens(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into chunks based on token count using simple iteration
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[Dict]: List of chunk dictionaries with text, tokens, and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        
        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        start_pos = 0
        
        # Process sentences one by one using simple for loop
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_tokens and current_chunk:
                # Create chunk
                chunk = self._create_chunk(current_chunk, chunk_id, start_pos)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_id += 1
                start_pos = len(text) - len(current_chunk)  # Approximate position
                
            else:
                # Add sentence to current chunk
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if there's remaining text
        if current_chunk.strip():
            chunk = self._create_chunk(current_chunk, chunk_id, start_pos)
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunk boundaries
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, and question marks
        # Keep the punctuation with the sentence
        sentences = re.split(r'([.!?]+)', text)
        
        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i]
            if i+1 < len(sentences):
                sentence += sentences[i+1]
            if sentence.strip():
                result.append(sentence + " ")
        
        return result
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of current chunk
        
        Args:
            text: Current chunk text
            
        Returns:
            str: Overlap text
        """
        if not text:
            return ""
        
        # Get last part of text for overlap (approximately overlap_tokens worth)
        words = text.split()
        overlap_words = min(len(words), self.overlap_tokens // 4)  # Rough estimate
        
        return " ".join(words[-overlap_words:]) + " " if overlap_words > 0 else ""
    
    def _create_chunk(self, text: str, chunk_id: int, start_pos: int) -> Dict[str, any]:
        """
        Create a chunk dictionary with metadata
        
        Args:
            text: Chunk text
            chunk_id: Unique chunk identifier
            start_pos: Starting position in document
            
        Returns:
            Dict: Chunk with metadata
        """
        text = text.strip()
        
        return {
            'id': chunk_id,
            'text': text,
            'start_position': start_pos,
            'end_position': start_pos + len(text),
            'length': len(text),
            'word_count': len(text.split()),
            'token_count': self.count_tokens(text),
            'token_estimate': self.count_tokens(text)
        }
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Main chunking method - uses token-based chunking
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[Dict]: List of chunk dictionaries
        """
        return self.chunk_text_by_tokens(text)
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics about the chunked text including token information
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dict: Statistics about chunks
        """
        if not chunks:
            return {}
        
        total_chars = sum(chunk['length'] for chunk in chunks)
        total_words = sum(chunk['word_count'] for chunk in chunks)
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'total_tokens': total_tokens,
            'average_chunk_size': total_chars / len(chunks) if chunks else 0,
            'average_words_per_chunk': total_words / len(chunks) if chunks else 0,
            'average_tokens_per_chunk': total_tokens / len(chunks) if chunks else 0,
            'max_tokens_per_chunk': max(chunk.get('token_count', 0) for chunk in chunks) if chunks else 0,
            'min_tokens_per_chunk': min(chunk.get('token_count', 0) for chunk in chunks) if chunks else 0
        }
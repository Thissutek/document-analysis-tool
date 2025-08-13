"""
Text Chunker Module
Handles document segmentation with simple for-loop chunking
"""
from typing import List, Dict
import streamlit as st


class TextChunker:
    """Segments documents into manageable chunks for analysis"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks using simple for loop
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[Dict]: List of chunk dictionaries with text, position, and metadata
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        text_length = len(text)
        
        # Simple for-loop chunking as specified in requirements
        start_pos = 0
        chunk_id = 0
        
        for start_pos in range(0, text_length, self.chunk_size - self.overlap):
            end_pos = min(start_pos + self.chunk_size, text_length)
            
            # Extract chunk text
            chunk_text = text[start_pos:end_pos]
            
            # Skip very short chunks at the end
            if len(chunk_text.strip()) < 50:
                break
            
            # Create chunk dictionary
            chunk = {
                'id': chunk_id,
                'text': chunk_text.strip(),
                'start_position': start_pos,
                'end_position': end_pos,
                'length': len(chunk_text.strip()),
                'word_count': len(chunk_text.strip().split())
            }
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Break if we've reached the end
            if end_pos >= text_length:
                break
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics about the chunked text
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dict: Statistics about chunks
        """
        if not chunks:
            return {}
        
        total_chars = sum(chunk['length'] for chunk in chunks)
        total_words = sum(chunk['word_count'] for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chunk_size': total_chars / len(chunks) if chunks else 0,
            'average_words_per_chunk': total_words / len(chunks) if chunks else 0
        }
"""
Document Parser Module
Handles PDF text extraction using PyPDF2
"""
import PyPDF2
from typing import Optional
import streamlit as st


class DocumentParser:
    """Extracts text from PDF documents"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, uploaded_file) -> Optional[str]:
        """
        Extract text from uploaded PDF file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text content or None if extraction fails
        """
        try:
            # Read PDF using PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = ""
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            # Basic text cleaning
            text_content = self._clean_text(text_content)
            
            return text_content
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and format extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize line breaks
        text = " ".join(text.split())
        
        # Add back some structure
        text = text.replace(". ", ".\n")
        
        return text
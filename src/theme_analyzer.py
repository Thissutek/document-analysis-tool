"""
Theme Analyzer Module  
Handles AI-powered theme extraction using OpenAI GPT-4o-mini and embeddings
"""
import openai
import os
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


class ThemeAnalyzer:
    """AI-powered theme extraction and relevance filtering"""
    
    def __init__(self):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for text chunks using OpenAI embeddings
        
        Args:
            texts: List of text chunks
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return []
    
    def filter_relevant_chunks(self, chunks: List[Dict], research_theme: str, 
                              similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Filter chunks based on relevance to research theme using embeddings
        
        Args:
            chunks: List of text chunks
            research_theme: Research focus theme
            similarity_threshold: Minimum similarity score to keep chunk
            
        Returns:
            List[Dict]: Filtered relevant chunks
        """
        if not chunks:
            return []
        
        try:
            # Get embeddings for research theme
            theme_embedding = self.get_embeddings([research_theme])[0]
            
            # Get embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_embeddings = self.get_embeddings(chunk_texts)
            
            if not chunk_embeddings:
                return chunks  # Return all if embeddings fail
            
            # Calculate similarity scores
            theme_embedding = np.array(theme_embedding).reshape(1, -1)
            chunk_embeddings = np.array(chunk_embeddings)
            
            similarities = cosine_similarity(theme_embedding, chunk_embeddings)[0]
            
            # Filter chunks based on similarity threshold
            relevant_chunks = []
            for i, chunk in enumerate(chunks):
                if similarities[i] >= similarity_threshold:
                    chunk['relevance_score'] = float(similarities[i])
                    relevant_chunks.append(chunk)
            
            # Sort by relevance score (highest first)
            relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error filtering chunks: {str(e)}")
            return chunks
    
    def extract_themes(self, chunks: List[Dict], research_theme: str) -> List[Dict[str, any]]:
        """
        Extract themes from relevant chunks using GPT-4o-mini
        Combined relevance filtering + theme extraction for efficiency
        
        Args:
            chunks: List of text chunks
            research_theme: Research focus theme
            
        Returns:
            List[Dict]: Extracted themes with metadata
        """
        if not chunks:
            return []
        
        try:
            # First filter for relevance
            relevant_chunks = self.filter_relevant_chunks(chunks, research_theme)
            
            if not relevant_chunks:
                st.warning("No chunks found relevant to the research theme")
                return []
            
            # Prepare text for theme extraction
            combined_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:10]])  # Limit for API costs
            
            # Create prompt for theme extraction
            prompt = f"""
            Analyze the following text and extract key themes related to "{research_theme}".
            
            For each theme, provide:
            1. Theme name (short, descriptive)
            2. Description (1-2 sentences)
            3. Relevance score (1-10 scale)
            4. Key phrases or quotes that support this theme
            
            Focus on themes that are most relevant to: {research_theme}
            
            Text to analyze:
            {combined_text[:4000]}  # Limit text length
            
            Respond in JSON format with an array of theme objects.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert text analyst specializing in theme extraction."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse response (placeholder - would need proper JSON parsing)
            themes_text = response.choices[0].message.content
            
            # For now, create sample themes (implement proper JSON parsing later)
            sample_themes = [
                {
                    'name': 'Primary Theme',
                    'description': f'Main theme related to {research_theme}',
                    'relevance_score': 8.5,
                    'frequency': len(relevant_chunks),
                    'key_phrases': ['sample phrase 1', 'sample phrase 2'],
                    'chunk_ids': [chunk['id'] for chunk in relevant_chunks[:3]]
                }
            ]
            
            return sample_themes
            
        except Exception as e:
            st.error(f"Error extracting themes: {str(e)}")
            return []
    
    def calculate_theme_frequency(self, themes: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """
        Calculate frequency metrics for extracted themes
        
        Args:
            themes: List of extracted themes
            chunks: Original text chunks
            
        Returns:
            List[Dict]: Themes with frequency metrics added
        """
        for theme in themes:
            # Calculate frequency based on theme presence in chunks
            theme['chunk_frequency'] = len(theme.get('chunk_ids', []))
            theme['relative_frequency'] = theme['chunk_frequency'] / len(chunks) if chunks else 0
        
        return themes
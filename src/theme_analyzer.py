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
        """Initialize with OpenAI API key if available"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            try:
                self.client = openai.OpenAI(api_key=api_key)
                self.has_api_key = True
            except Exception as e:
                st.warning(f"OpenAI client initialization failed: {e}")
                self.client = None
                self.has_api_key = False
        else:
            self.client = None
            self.has_api_key = False
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for text chunks using OpenAI embeddings
        
        Args:
            texts: List of text chunks
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not self.has_api_key or not self.client:
            return []
            
        try:
            response = self.client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return []
    
    def filter_relevant_chunks_ai(self, chunks: List[Dict], research_topics: List[str], 
                                 similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Filter chunks based on relevance to research topics using AI embeddings
        
        Args:
            chunks: List of text chunks
            research_topics: List of research topics/themes to search for
            similarity_threshold: Minimum similarity score to keep chunk
            
        Returns:
            List[Dict]: Filtered relevant chunks with relevance scores
        """
        if not chunks or not research_topics:
            return chunks
        
        try:
            # AI analysis in progress - status handled by main app
            
            # Combine research topics into a single query for embedding
            combined_topics = " ".join(research_topics)
            
            # Get embeddings for combined research topics
            topic_embeddings = self.get_embeddings([combined_topics])
            if not topic_embeddings:
                st.warning("Failed to get topic embeddings, using fallback method")
                return self._fallback_keyword_filter(chunks, research_topics)
            
            topic_embedding = topic_embeddings[0]
            
            # Process chunks in batches to avoid API limits
            batch_size = 10
            relevant_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                chunk_texts = [chunk['text'] for chunk in batch_chunks]
                
                # Get embeddings for chunk batch
                chunk_embeddings = self.get_embeddings(chunk_texts)
                
                if not chunk_embeddings:
                    st.warning(f"Failed to get embeddings for batch {i//batch_size + 1}")
                    continue
                
                # Calculate similarity scores
                topic_embedding_array = np.array(topic_embedding).reshape(1, -1)
                chunk_embeddings_array = np.array(chunk_embeddings)
                
                similarities = cosine_similarity(topic_embedding_array, chunk_embeddings_array)[0]
                
                # Filter chunks based on similarity threshold
                for j, chunk in enumerate(batch_chunks):
                    if similarities[j] >= similarity_threshold:
                        chunk['relevance_score'] = float(similarities[j])
                        chunk['relevance_method'] = 'ai_embedding'
                        relevant_chunks.append(chunk)
                
                # Progress tracking handled by main app
            
            # Sort by relevance score (highest first)
            relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Analysis complete - detailed results will be shown in main app
            return relevant_chunks
            
        except Exception as e:
            st.warning(f"AI filtering failed: {str(e)}. Using fallback method.")
            return self._fallback_keyword_filter(chunks, research_topics)
    
    def _fallback_keyword_filter(self, chunks: List[Dict], research_topics: List[str]) -> List[Dict]:
        """
        Fallback method using keyword matching when AI embeddings fail
        
        Args:
            chunks: List of text chunks
            research_topics: List of research topics
            
        Returns:
            List[Dict]: Filtered chunks using keyword matching
        """
        relevant_chunks = []
        
        for chunk in chunks:
            chunk_text_lower = chunk['text'].lower()
            
            # Calculate relevance based on keyword matches
            matches = 0
            total_topics = len(research_topics)
            
            for topic in research_topics:
                topic_words = topic.lower().split()
                # Check if any words from the topic appear in the chunk
                if any(word in chunk_text_lower for word in topic_words if len(word) > 3):
                    matches += 1
            
            # Calculate relevance score
            if matches > 0:
                relevance_score = matches / total_topics
                chunk['relevance_score'] = relevance_score
                chunk['relevance_method'] = 'keyword_matching'
                relevant_chunks.append(chunk)
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_chunks
    
    def filter_relevant_chunks(self, chunks: List[Dict], research_topics: List[str], 
                              similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Main method to filter chunks - tries AI first, falls back to keywords
        
        Args:
            chunks: List of text chunks
            research_topics: List of research topics/themes
            similarity_threshold: Minimum similarity score to keep chunk
            
        Returns:
            List[Dict]: Filtered relevant chunks
        """
        # Try AI-powered filtering first
        return self.filter_relevant_chunks_ai(chunks, research_topics, similarity_threshold)
    
    def extract_themes_from_chunks(self, relevant_chunks: List[Dict], research_topics: List[str], max_themes: int = 15) -> List[Dict[str, any]]:
        """
        Extract themes from relevant chunks using GPT-4o-mini
        
        Args:
            relevant_chunks: List of filtered relevant chunks
            research_topics: List of research topics to focus on
            max_themes: Maximum number of themes to extract
            
        Returns:
            List[Dict]: Extracted themes with metadata
        """
        if not relevant_chunks:
            return []
        
        if not self.has_api_key:
            st.warning("OpenAI API not available. Using fallback theme extraction.")
            return self._fallback_theme_extraction(relevant_chunks, research_topics, max_themes)
        
        try:
            # Theme extraction in progress - status handled by main app
            
            # Process chunks in smaller batches for better theme extraction
            all_themes = []
            batch_size = 5  # Smaller batches for more focused analysis
            
            combined_topics = ", ".join(research_topics)
            
            for i in range(0, min(len(relevant_chunks), 20), batch_size):  # Limit to first 20 chunks
                batch_chunks = relevant_chunks[i:i + batch_size]
                batch_text = "\n\n---CHUNK BREAK---\n\n".join([f"Chunk {chunk['id']}: {chunk['text']}" for chunk in batch_chunks])
                
                # Create focused prompt for theme extraction
                prompt = f"""
Analyze the following text chunks and extract distinct themes related to these research topics: {combined_topics}

For each theme you identify, provide:
1. A clear, specific theme name (2-5 words)
2. A brief description (1 sentence)
3. Evidence from the text (key phrases or quotes)
4. Which chunks contain this theme (by chunk ID)

Focus on finding {max_themes//3} distinct themes from this batch.
Return ONLY a valid JSON array with this structure:
[
  {{
    "name": "Theme Name",
    "description": "Brief description of the theme",
    "evidence": ["key phrase 1", "key phrase 2"],
    "chunk_ids": [1, 3, 5],
    "confidence": 0.85
  }}
]

Text to analyze:
{batch_text[:3000]}
"""
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert researcher who extracts themes from academic and business texts. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                
                # Parse JSON response
                try:
                    batch_themes = self._parse_theme_response(response.choices[0].message.content, batch_chunks)
                    all_themes.extend(batch_themes)
                except Exception as e:
                    st.warning(f"Failed to parse themes from batch {i//batch_size + 1}: {e}")
                
                # Progress tracking handled by main app
            
            # Deduplicate and consolidate themes
            consolidated_themes = self._consolidate_themes(all_themes, max_themes)
            
            # Theme extraction complete - results will be displayed in main app
            return consolidated_themes
            
        except Exception as e:
            st.warning(f"GPT theme extraction failed: {str(e)}. Using fallback method.")
            return self._fallback_theme_extraction(relevant_chunks, research_topics, max_themes)
    
    def _parse_theme_response(self, response_text: str, chunks: List[Dict]) -> List[Dict]:
        """Parse GPT response and create theme objects"""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                themes_data = json.loads(json_match.group())
            else:
                themes_data = json.loads(response_text)
            
            themes = []
            for theme_data in themes_data:
                if isinstance(theme_data, dict) and 'name' in theme_data:
                    theme = {
                        'name': theme_data.get('name', 'Unknown Theme'),
                        'description': theme_data.get('description', 'No description available'),
                        'evidence': theme_data.get('evidence', []),
                        'chunk_ids': theme_data.get('chunk_ids', []),
                        'confidence': theme_data.get('confidence', 0.5),
                        'chunk_frequency': len(theme_data.get('chunk_ids', [])),
                        'source': 'gpt-4o-mini'
                    }
                    themes.append(theme)
            
            return themes
            
        except Exception as e:
            st.warning(f"JSON parsing failed: {e}")
            return []
    
    def _consolidate_themes(self, themes: List[Dict], max_themes: int) -> List[Dict]:
        """Consolidate similar themes and limit to max_themes"""
        if not themes:
            return []
        
        # Simple deduplication by name similarity
        unique_themes = []
        theme_names = set()
        
        for theme in themes:
            theme_name_lower = theme['name'].lower()
            
            # Check if similar theme already exists
            is_duplicate = any(
                self._themes_similar(theme_name_lower, existing.lower()) 
                for existing in theme_names
            )
            
            if not is_duplicate:
                theme_names.add(theme['name'])
                unique_themes.append(theme)
        
        # Sort by confidence and limit
        unique_themes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return unique_themes[:max_themes]
    
    def _themes_similar(self, theme1: str, theme2: str) -> bool:
        """Check if two theme names are similar"""
        # Simple similarity check - can be improved
        words1 = set(theme1.split())
        words2 = set(theme2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        min_length = min(len(words1), len(words2))
        
        return overlap / min_length > 0.6  # 60% word overlap
    
    def _fallback_theme_extraction(self, chunks: List[Dict], research_topics: List[str], max_themes: int) -> List[Dict]:
        """Fallback theme extraction using keyword analysis"""
        themes = []
        
        # Create themes based on research topics and chunk analysis
        for i, topic in enumerate(research_topics[:max_themes]):
            relevant_chunks = [
                chunk for chunk in chunks 
                if any(word.lower() in chunk['text'].lower() for word in topic.split() if len(word) > 3)
            ]
            
            if relevant_chunks:
                # Extract key phrases from relevant chunks
                key_phrases = self._extract_key_phrases(relevant_chunks, topic)
                
                theme = {
                    'name': topic.title(),
                    'description': f'Theme related to {topic} found in document analysis',
                    'evidence': key_phrases[:3],  # Top 3 key phrases
                    'chunk_ids': [chunk['id'] for chunk in relevant_chunks[:5]],
                    'confidence': len(relevant_chunks) / len(chunks),
                    'chunk_frequency': len(relevant_chunks),
                    'source': 'keyword_analysis'
                }
                themes.append(theme)
        
        return themes[:max_themes]
    
    def _extract_key_phrases(self, chunks: List[Dict], topic: str) -> List[str]:
        """Extract key phrases related to topic from chunks"""
        phrases = []
        topic_words = [word.lower() for word in topic.split() if len(word) > 3]
        
        for chunk in chunks[:3]:  # Look at first few chunks
            text = chunk['text']
            sentences = text.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word in sentence.lower() for word in topic_words):
                    phrases.append(sentence[:100])  # First 100 chars
        
        return phrases[:5]  # Return up to 5 phrases
    
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
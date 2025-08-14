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
        Extract themes from relevant chunks using enhanced GPT-4o-mini prompting
        
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
            # Stage 1: Enhanced initial theme extraction
            initial_themes = self._enhanced_theme_extraction(relevant_chunks, research_topics, max_themes)
            
            # Stage 2: Validation and refinement
            if initial_themes:
                validated_themes = self._validate_and_refine_themes(initial_themes, relevant_chunks, research_topics)
                return validated_themes
            else:
                return initial_themes
            
        except Exception as e:
            st.warning(f"Enhanced GPT theme extraction failed: {str(e)}. Using fallback method.")
            return self._fallback_theme_extraction(relevant_chunks, research_topics, max_themes)
    
    def _enhanced_theme_extraction(self, relevant_chunks: List[Dict], research_topics: List[str], max_themes: int) -> List[Dict]:
        """Stage 1: Enhanced theme extraction with improved prompting"""
        all_themes = []
        batch_size = 4  # Smaller batches for more focused analysis
        
        combined_topics = ", ".join(research_topics)
        themes_per_batch = max(2, max_themes // max(1, len(relevant_chunks) // batch_size))
        
        for i in range(0, min(len(relevant_chunks), 16), batch_size):  # Focus on most relevant chunks
            batch_chunks = relevant_chunks[i:i + batch_size]
            batch_text = self._prepare_context_text(batch_chunks)
            
            # Enhanced prompt with specific instructions
            prompt = f"""
You are a senior qualitative researcher conducting thematic analysis on research data.

RESEARCH CONTEXT:
- Primary research focus: {combined_topics}
- Document type: Academic/business analysis
- Analysis goal: Identify actionable, specific themes

TASK: Extract {themes_per_batch} DISTINCT, SPECIFIC themes from the provided text chunks.

CRITICAL REQUIREMENTS:
1. SPECIFICITY: Avoid generic themes. Instead of "communication," use "cross-team communication barriers"
2. EVIDENCE-BASED: Each theme MUST be supported by concrete text evidence
3. ACTIONABILITY: Themes should be specific enough to guide decision-making
4. RELIABILITY: Each theme should appear across multiple chunks when possible
5. RESEARCH ALIGNMENT: Themes must relate to the research focus: {combined_topics}

QUALITY STANDARDS:
- Theme names: 2-6 words, descriptive and specific
- Evidence: Direct quotes or specific phrases from the text
- Confidence: Base on evidence strength and chunk support (0.3-1.0)

OUTPUT FORMAT: Valid JSON array only
[
  {{
    "name": "Specific Theme Name",
    "description": "One clear sentence explaining this theme's significance to your research",
    "evidence": ["direct quote from text", "specific phrase showing this theme"],
    "chunk_ids": [1, 3],
    "confidence": 0.85,
    "justification": "Why this theme is important and well-supported"
  }}
]

TEXT TO ANALYZE:
{batch_text}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert qualitative researcher specializing in thematic analysis. 
Your expertise includes:
- Identifying specific, actionable themes from complex texts
- Distinguishing between surface-level and deeper thematic patterns
- Ensuring themes are grounded in evidence and relevant to research objectives
- Maintaining high standards for theme quality and specificity

Always respond with valid JSON only. No explanations or additional text."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=1200   # More tokens for detailed analysis
            )
            
            # Parse JSON response
            try:
                batch_themes = self._parse_enhanced_theme_response(response.choices[0].message.content, batch_chunks)
                all_themes.extend(batch_themes)
            except Exception as e:
                st.warning(f"Failed to parse themes from batch {i//batch_size + 1}: {e}")
        
        return all_themes
    
    def _validate_and_refine_themes(self, initial_themes: List[Dict], chunks: List[Dict], research_topics: List[str]) -> List[Dict]:
        """Stage 2: Validation and refinement of extracted themes"""
        if not initial_themes:
            return initial_themes
        
        # Prepare themes for validation
        themes_summary = []
        for theme in initial_themes:
            themes_summary.append({
                'name': theme['name'],
                'description': theme['description'],
                'evidence_count': len(theme.get('evidence', [])),
                'chunk_count': len(theme.get('chunk_ids', [])),
                'confidence': theme.get('confidence', 0.5)
            })
        
        combined_topics = ", ".join(research_topics)
        
        # Validation prompt
        validation_prompt = f"""
You are reviewing thematic analysis results for quality and relevance.

ORIGINAL RESEARCH FOCUS: {combined_topics}

EXTRACTED THEMES TO VALIDATE:
{self._format_themes_for_validation(themes_summary)}

VALIDATION CRITERIA:
1. RELEVANCE: Does each theme directly relate to the research focus?
2. SPECIFICITY: Is the theme specific enough to be actionable?
3. EVIDENCE QUALITY: Is there sufficient evidence support?
4. DISTINCTIVENESS: Are themes truly distinct from each other?
5. RESEARCH VALUE: Would this theme provide valuable insights?

TASKS:
1. Identify any themes that should be MERGED (too similar)
2. Identify any themes that should be REMOVED (insufficient evidence, irrelevant, too generic)
3. Suggest IMPROVEMENTS to theme names or descriptions for clarity
4. Assign final QUALITY SCORES (0.0-1.0) based on validation criteria

OUTPUT FORMAT: Valid JSON only
{{
  "validated_themes": [
    {{
      "original_name": "Theme Name",
      "action": "keep|merge|remove|improve",
      "merge_with": "Other Theme Name", 
      "improved_name": "Better Theme Name",
      "improved_description": "Enhanced description", 
      "quality_score": 0.85,
      "reasoning": "Brief explanation of decision"
    }}
  ],
  "overall_assessment": "Brief summary of theme quality and coverage"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior research methodologist specializing in qualitative analysis validation. Provide only valid JSON responses."
                    },
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Apply validation results
            return self._apply_validation_results(initial_themes, response.choices[0].message.content)
            
        except Exception as e:
            st.warning(f"Theme validation failed: {e}. Returning initial themes.")
            # Apply basic consolidation as fallback
            return self._consolidate_themes(initial_themes, len(initial_themes))
    
    def _prepare_context_text(self, chunks: List[Dict], max_length: int = 3500) -> str:
        """Prepare optimized context text from chunks"""
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            chunk_header = f"\n=== CHUNK {chunk['id']} ===\n"
            chunk_text = chunk['text']
            
            # Calculate if this chunk fits
            chunk_content = chunk_header + chunk_text + "\n"
            
            if current_length + len(chunk_content) <= max_length:
                context_parts.append(chunk_content)
                current_length += len(chunk_content)
            else:
                # Try to fit partial chunk
                remaining_space = max_length - current_length - len(chunk_header) - 10
                if remaining_space > 200:  # Only if substantial space
                    partial_text = chunk_text[:remaining_space] + "..."
                    context_parts.append(chunk_header + partial_text + "\n")
                break
        
        return "".join(context_parts)
    
    def _format_themes_for_validation(self, themes: List[Dict]) -> str:
        """Format themes for validation prompt"""
        formatted = []
        for i, theme in enumerate(themes, 1):
            formatted.append(
                f"{i}. {theme['name']}\n"
                f"   Description: {theme['description']}\n"
                f"   Evidence: {theme['evidence_count']} pieces\n"
                f"   Chunks: {theme['chunk_count']}\n"
                f"   Confidence: {theme['confidence']:.2f}\n"
            )
        return "\n".join(formatted)
    
    def _parse_enhanced_theme_response(self, response_text: str, chunks: List[Dict]) -> List[Dict]:
        """Parse enhanced GPT response with additional fields"""
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
                        'source': 'gpt-4o-mini-enhanced',
                        'justification': theme_data.get('justification', ''),
                        'extraction_method': 'enhanced_prompting'
                    }
                    themes.append(theme)
            
            return themes
            
        except Exception as e:
            st.warning(f"Enhanced JSON parsing failed: {e}")
            return []
    
    def _apply_validation_results(self, initial_themes: List[Dict], validation_response: str) -> List[Dict]:
        """Apply validation results to refine themes"""
        import json
        import re
        
        try:
            # Parse validation response
            json_match = re.search(r'\{.*\}', validation_response, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
            else:
                validation_data = json.loads(validation_response)
            
            validated_themes = []
            theme_lookup = {theme['name']: theme for theme in initial_themes}
            
            for validation in validation_data.get('validated_themes', []):
                original_name = validation.get('original_name')
                action = validation.get('action', 'keep')
                
                if original_name in theme_lookup:
                    theme = theme_lookup[original_name].copy()
                    
                    if action == 'keep':
                        # Update quality score
                        theme['validation_score'] = validation.get('quality_score', theme.get('confidence', 0.5))
                        validated_themes.append(theme)
                        
                    elif action == 'improve':
                        # Apply improvements
                        if validation.get('improved_name'):
                            theme['name'] = validation['improved_name']
                        if validation.get('improved_description'):
                            theme['description'] = validation['improved_description']
                        theme['validation_score'] = validation.get('quality_score', theme.get('confidence', 0.5))
                        theme['improvement_applied'] = True
                        validated_themes.append(theme)
                        
                    elif action == 'merge':
                        # Handle merge logic (simplified for now)
                        theme['merge_candidate'] = validation.get('merge_with')
                        theme['validation_score'] = validation.get('quality_score', theme.get('confidence', 0.5))
                        validated_themes.append(theme)
                    
                    # action == 'remove' means don't add to validated_themes
            
            # Sort by validation score and return
            validated_themes.sort(key=lambda x: x.get('validation_score', 0), reverse=True)
            return validated_themes
            
        except Exception as e:
            st.warning(f"Failed to apply validation results: {e}")
            # Return top themes by confidence as fallback
            initial_themes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            return initial_themes
    
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
    
    def calculate_theme_relevance_scores(self, themes: List[Dict], research_topics: List[str], 
                                       research_questions: List[str] = None) -> List[Dict]:
        """
        Calculate relevance scores for themes based on user's research topics and questions
        
        Args:
            themes: List of extracted themes
            research_topics: List of user-provided research topics
            research_questions: List of user-provided research questions
            
        Returns:
            List[Dict]: Themes with calculated relevance scores
        """
        if not research_topics and not research_questions:
            # If no user inputs, use default scores
            for theme in themes:
                theme['relevance_score'] = 5.0  # Default middle score
            return themes
        
        # Combine all user inputs for analysis
        all_user_inputs = research_topics.copy() if research_topics else []
        if research_questions:
            all_user_inputs.extend(research_questions)
        
        for theme in themes:
            theme_name = theme['name'].lower()
            theme_description = theme.get('description', '').lower()
            theme_evidence = ' '.join([e.lower() for e in theme.get('evidence', [])])
            
            # Calculate relevance based on multiple factors
            relevance_scores = []
            
            # 1. Direct topic matching
            topic_matches = 0
            for topic in research_topics:
                topic_words = [word.lower() for word in topic.split() if len(word) > 2]
                # Check if any topic words appear in theme name, description, or evidence
                for word in topic_words:
                    if (word in theme_name or 
                        word in theme_description or 
                        word in theme_evidence):
                        topic_matches += 1
                        break  # Count each topic only once
            
            topic_relevance = topic_matches / len(research_topics) if research_topics else 0
            relevance_scores.append(topic_relevance * 10)  # Scale to 0-10
            
            # 2. Question relevance (if questions provided)
            if research_questions:
                question_matches = 0
                for question in research_questions:
                    question_words = [word.lower() for word in question.split() if len(word) > 2]
                    # Check if question words appear in theme
                    for word in question_words:
                        if (word in theme_name or 
                            word in theme_description or 
                            word in theme_evidence):
                            question_matches += 1
                            break
                
                question_relevance = question_matches / len(research_questions)
                relevance_scores.append(question_relevance * 10)
            
            # 3. Semantic similarity using embeddings (if available)
            if self.has_api_key and all_user_inputs:
                try:
                    # Get embeddings for theme and user inputs
                    theme_text = f"{theme_name} {theme_description}"
                    combined_inputs = " ".join(all_user_inputs)
                    
                    embeddings = self.get_embeddings([theme_text, combined_inputs])
                    if len(embeddings) == 2:
                        # Calculate cosine similarity
                        theme_embedding = np.array(embeddings[0])
                        inputs_embedding = np.array(embeddings[1])
                        
                        similarity = np.dot(theme_embedding, inputs_embedding) / (
                            np.linalg.norm(theme_embedding) * np.linalg.norm(inputs_embedding)
                        )
                        
                        # Scale similarity to 0-10 range
                        semantic_relevance = max(0, similarity * 10)
                        relevance_scores.append(semantic_relevance)
                except Exception as e:
                    # If embedding fails, continue without it
                    pass
            
            # 4. Confidence score from theme extraction
            confidence_score = theme.get('confidence', 0.5) * 10  # Scale to 0-10
            relevance_scores.append(confidence_score)
            
            # Calculate weighted average relevance score
            if len(relevance_scores) > 0:
                # Weight topic matching more heavily
                weights = [0.4, 0.3, 0.2, 0.1]  # Adjust based on number of scores
                if len(relevance_scores) == 1:
                    weights = [1.0]
                elif len(relevance_scores) == 2:
                    weights = [0.6, 0.4]
                elif len(relevance_scores) == 3:
                    weights = [0.5, 0.3, 0.2]
                
                # Ensure weights sum to 1
                weights = weights[:len(relevance_scores)]
                weights = [w / sum(weights) for w in weights]
                
                final_relevance = sum(score * weight for score, weight in zip(relevance_scores, weights))
                theme['relevance_score'] = min(10.0, max(0.0, final_relevance))
            else:
                theme['relevance_score'] = 5.0  # Default score
        
        return themes
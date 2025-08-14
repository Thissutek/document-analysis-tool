"""
Relationship Calculator Module
Handles theme correlation analysis and co-occurrence counting
"""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st


class RelationshipCalculator:
    """Calculate theme relationships and correlations"""
    
    def __init__(self):
        pass
    
    def calculate_theme_relationships(self, themes: List[Dict], all_chunks: List[Dict], 
                                    research_topics: List[str] = None, 
                                    research_questions: List[str] = None) -> Dict[str, any]:
        """
        Calculate comprehensive relationships between themes
        
        Args:
            themes: List of extracted themes with chunk_ids
            all_chunks: All document chunks for context
            research_topics: List of user-provided research topics
            research_questions: List of user-provided research questions
            
        Returns:
            Dict: Complete relationship analysis including co-occurrence, correlations, and metrics
        """
        if not themes:
            return {'cooccurrence': {}, 'correlations': {}, 'theme_metrics': {}}
        
        # Relationship calculation in progress - status handled by main app
        
        # Step 1: Calculate co-occurrence matrix
        cooccurrence = self.calculate_cooccurrence(themes, all_chunks)
        
        # Step 2: Calculate correlation strengths
        correlations = self.calculate_correlation_strength(themes, cooccurrence)
        
        # Step 3: Calculate theme centrality and importance
        theme_metrics = self.calculate_theme_metrics(themes, correlations)
        
        # Step 4: Calculate research focus relationships based on user inputs
        research_relationships = self.measure_research_focus_relationship(
            themes, research_topics, research_questions
        )
        
        return {
            'cooccurrence': cooccurrence,
            'correlations': correlations,
            'theme_metrics': theme_metrics,
            'research_relationships': research_relationships,
            'total_themes': len(themes)
        }
    
    def calculate_cooccurrence(self, themes: List[Dict], chunks: List[Dict]) -> Dict[Tuple[str, str], int]:
        """
        Calculate co-occurrence counts between themes
        
        Args:
            themes: List of extracted themes
            chunks: Text chunks for context
            
        Returns:
            Dict: Co-occurrence counts for theme pairs
        """
        cooccurrence = {}
        
        # Create theme-chunk mapping from chunk_ids in themes
        theme_chunks = {}
        for theme in themes:
            theme_name = theme['name']
            chunk_ids = theme.get('chunk_ids', [])
            theme_chunks[theme_name] = set(chunk_ids)
        
        # Calculate co-occurrence for each theme pair
        theme_names = list(theme_chunks.keys())
        
        for i, theme1 in enumerate(theme_names):
            for j, theme2 in enumerate(theme_names):
                if i < j:  # Avoid duplicates and self-comparison
                    # Count chunks where both themes appear
                    intersection = theme_chunks[theme1].intersection(theme_chunks[theme2])
                    cooccurrence[(theme1, theme2)] = len(intersection)
        
        return cooccurrence
    
    def calculate_correlation_strength(self, themes: List[Dict], cooccurrence: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
        """
        Calculate correlation strength between theme pairs using multiple methods
        
        Args:
            themes: List of themes with frequency data
            cooccurrence: Co-occurrence counts
            
        Returns:
            Dict: Correlation strengths (0-1 scale)
        """
        correlations = {}
        
        # Create theme frequency mapping
        theme_freq = {theme['name']: theme.get('chunk_frequency', 1) for theme in themes}
        
        for (theme1, theme2), cooccur_count in cooccurrence.items():
            freq1 = theme_freq.get(theme1, 1)
            freq2 = theme_freq.get(theme2, 1)
            
            if freq1 == 0 or freq2 == 0:
                correlations[(theme1, theme2)] = 0.0
                continue
            
            # Calculate multiple correlation measures and average them
            
            # 1. Jaccard similarity (intersection over union)
            union_size = freq1 + freq2 - cooccur_count
            jaccard = cooccur_count / union_size if union_size > 0 else 0.0
            
            # 2. Conditional probability (P(theme2|theme1) + P(theme1|theme2)) / 2
            conditional_prob = (cooccur_count / freq1 + cooccur_count / freq2) / 2
            
            # 3. Cosine similarity approximation
            cosine = cooccur_count / (freq1 * freq2) ** 0.5 if freq1 > 0 and freq2 > 0 else 0.0
            
            # Average the measures with weights
            correlation = (0.4 * jaccard + 0.4 * conditional_prob + 0.2 * cosine)
            correlation = min(1.0, correlation)  # Cap at 1.0
            
            correlations[(theme1, theme2)] = correlation
        
        return correlations
    
    def calculate_theme_metrics(self, themes: List[Dict], correlations: Dict[Tuple[str, str], float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate centrality and importance metrics for each theme
        
        Args:
            themes: List of themes
            correlations: Theme correlation strengths
            
        Returns:
            Dict: Metrics for each theme including centrality, importance, etc.
        """
        theme_metrics = {}
        theme_names = [theme['name'] for theme in themes]
        
        for theme in themes:
            theme_name = theme['name']
            
            # Calculate centrality (how connected this theme is to others)
            connections = []
            for (t1, t2), strength in correlations.items():
                if t1 == theme_name:
                    connections.append(strength)
                elif t2 == theme_name:
                    connections.append(strength)
            
            centrality = sum(connections) / len(theme_names) if theme_names else 0
            
            # Calculate importance (combination of frequency and centrality)
            frequency_score = theme.get('chunk_frequency', 0) / max(1, max(t.get('chunk_frequency', 1) for t in themes))
            confidence_score = theme.get('confidence', 0.5)
            
            importance = (0.4 * frequency_score + 0.3 * centrality + 0.3 * confidence_score)
            
            theme_metrics[theme_name] = {
                'centrality': centrality,
                'importance': importance,
                'frequency_score': frequency_score,
                'confidence_score': confidence_score,
                'connection_count': len(connections),
                'avg_connection_strength': sum(connections) / len(connections) if connections else 0
            }
        
        return theme_metrics
    
    def measure_research_focus_relationship(self, themes: List[Dict], research_topics: List[str] = None, 
                                          research_questions: List[str] = None) -> Dict[str, float]:
        """
        Measure how closely each theme relates to the user's research topics and questions
        
        Args:
            themes: List of extracted themes
            research_topics: List of user-provided research topics
            research_questions: List of user-provided research questions
            
        Returns:
            Dict: Research focus relationship scores for each theme
        """
        research_relationships = {}
        
        for theme in themes:
            # Use the relevance score that was calculated based on user inputs
            relevance = theme.get('relevance_score', 5.0)
            
            # Normalize to 0-1 scale (relevance_score is already 0-10)
            normalized_relevance = relevance / 10.0
            
            research_relationships[theme['name']] = normalized_relevance
        
        return research_relationships
    
    def create_relationship_matrix(self, themes: List[Dict]) -> pd.DataFrame:
        """
        Create a relationship matrix for visualization
        
        Args:
            themes: List of themes with relationship data
            
        Returns:
            pd.DataFrame: Relationship matrix
        """
        theme_names = [theme['name'] for theme in themes]
        
        if not theme_names:
            return pd.DataFrame()
        
        # Initialize matrix
        matrix = pd.DataFrame(
            np.zeros((len(theme_names), len(theme_names))),
            index=theme_names,
            columns=theme_names
        )
        
        # Fill diagonal with theme frequencies (normalized)
        max_freq = max([theme.get('chunk_frequency', 1) for theme in themes])
        
        for i, theme in enumerate(themes):
            freq = theme.get('chunk_frequency', 1)
            normalized_freq = freq / max_freq if max_freq > 0 else 0.5
            matrix.iloc[i, i] = normalized_freq
        
        return matrix
    
    def prepare_visualization_data(self, themes: List[Dict], relationship_analysis: Dict[str, any]) -> Dict[str, any]:
        """
        Prepare comprehensive data structure optimized for Streamlit/Plotly visualization
        
        Args:
            themes: List of extracted themes
            relationship_analysis: Complete relationship analysis from calculate_theme_relationships
            
        Returns:
            Dict: Visualization-ready data structure with nodes, edges, and metadata
        """
        if not themes:
            return {'nodes': [], 'edges': [], 'theme_count': 0, 'relationship_count': 0}
        
        correlations = relationship_analysis.get('correlations', {})
        theme_metrics = relationship_analysis.get('theme_metrics', {})
        research_relationships = relationship_analysis.get('research_relationships', {})
        
        # Prepare nodes (themes) data with comprehensive metrics
        nodes = []
        for theme in themes:
            theme_name = theme['name']
            metrics = theme_metrics.get(theme_name, {})
            
            node = {
                'id': theme_name,
                'label': theme_name,
                'description': theme.get('description', 'No description available'),
                'evidence': theme.get('evidence', []),
                'source': theme.get('source', 'unknown'),
                
                # Size and frequency data
                'size': theme.get('chunk_frequency', 1),
                'frequency': theme.get('chunk_frequency', 1),
                'chunk_ids': theme.get('chunk_ids', []),
                
                # Confidence and relationship scores
                'confidence': theme.get('confidence', 0.5),
                'research_relationship': research_relationships.get(theme_name, 0.5),
                
                # Advanced metrics
                'centrality': metrics.get('centrality', 0.0),
                'importance': metrics.get('importance', 0.5),
                'connection_count': metrics.get('connection_count', 0),
                'avg_connection_strength': metrics.get('avg_connection_strength', 0.0)
            }
            nodes.append(node)
        
        # Prepare edges (relationships) data
        edges = []
        for (theme1, theme2), strength in correlations.items():
            if strength > 0.05:  # Filter very weak relationships
                # Get co-occurrence count for additional context
                cooccurrence = relationship_analysis.get('cooccurrence', {})
                cooccur_count = cooccurrence.get((theme1, theme2), 0)
                
                edge = {
                    'source': theme1,
                    'target': theme2,
                    'strength': strength,
                    'weight': strength * 10,  # Scale for visualization
                    'cooccurrence_count': cooccur_count,
                    'relationship_type': self._classify_relationship_strength(strength)
                }
                edges.append(edge)
        
        # Calculate summary statistics
        max_frequency = max([node['frequency'] for node in nodes]) if nodes else 1
        avg_confidence = sum([node['confidence'] for node in nodes]) / len(nodes) if nodes else 0
        total_relationships = len(edges)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'theme_count': len(themes),
            'relationship_count': total_relationships,
            'max_frequency': max_frequency,
            'avg_confidence': avg_confidence,
            'strong_relationships': len([e for e in edges if e['strength'] > 0.3]),
            'summary_stats': {
                'total_themes': len(themes),
                'total_relationships': total_relationships,
                'avg_theme_confidence': avg_confidence,
                'max_theme_frequency': max_frequency,
                'themes_by_source': self._group_themes_by_source(themes)
            }
        }
    
    def _classify_relationship_strength(self, strength: float) -> str:
        """Classify relationship strength into categories"""
        if strength >= 0.6:
            return 'very_strong'
        elif strength >= 0.4:
            return 'strong'
        elif strength >= 0.2:
            return 'moderate'
        else:
            return 'weak'
    
    def _group_themes_by_source(self, themes: List[Dict]) -> Dict[str, int]:
        """Group themes by their extraction source"""
        sources = {}
        for theme in themes:
            source = theme.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources
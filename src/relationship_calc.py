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
    
    def calculate_cooccurrence(self, themes: List[Dict], chunks: List[Dict]) -> Dict[Tuple[str, str], int]:
        """
        Calculate co-occurrence counts between themes
        
        Args:
            themes: List of extracted themes
            chunks: Text chunks with theme associations
            
        Returns:
            Dict: Co-occurrence counts for theme pairs
        """
        cooccurrence = {}
        
        # Create theme-chunk mapping
        theme_chunks = {}
        for theme in themes:
            theme_name = theme['name']
            theme_chunks[theme_name] = set(theme.get('chunk_ids', []))
        
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
        Calculate correlation strength between theme pairs
        
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
            
            # Calculate Jaccard similarity as correlation strength
            union_size = freq1 + freq2 - cooccur_count
            if union_size > 0:
                correlation = cooccur_count / union_size
            else:
                correlation = 0.0
            
            correlations[(theme1, theme2)] = correlation
        
        return correlations
    
    def measure_research_focus_relationship(self, themes: List[Dict], research_theme: str) -> Dict[str, float]:
        """
        Measure how closely each theme relates to the main research focus
        
        Args:
            themes: List of extracted themes
            research_theme: Original research theme/focus
            
        Returns:
            Dict: Research focus relationship scores for each theme
        """
        research_relationships = {}
        
        for theme in themes:
            # Use existing relevance score or calculate based on theme description
            relevance = theme.get('relevance_score', 5.0)
            
            # Normalize to 0-1 scale
            normalized_relevance = relevance / 10.0 if relevance <= 10 else relevance
            
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
    
    def prepare_visualization_data(self, themes: List[Dict], correlations: Dict[Tuple[str, str], float], 
                                  research_relationships: Dict[str, float]) -> Dict[str, any]:
        """
        Prepare data structure optimized for Streamlit/Plotly visualization
        
        Args:
            themes: List of themes
            correlations: Theme-theme correlations
            research_relationships: Theme-research focus relationships
            
        Returns:
            Dict: Visualization-ready data structure
        """
        if not themes:
            return {}
        
        # Prepare nodes (themes) data
        nodes = []
        for theme in themes:
            node = {
                'id': theme['name'],
                'label': theme['name'],
                'description': theme.get('description', ''),
                'size': theme.get('chunk_frequency', 1),
                'frequency': theme.get('chunk_frequency', 1),
                'relevance_score': theme.get('relevance_score', 5.0),
                'research_relationship': research_relationships.get(theme['name'], 0.5)
            }
            nodes.append(node)
        
        # Prepare edges (relationships) data
        edges = []
        for (theme1, theme2), strength in correlations.items():
            if strength > 0.1:  # Filter weak relationships
                edge = {
                    'source': theme1,
                    'target': theme2,
                    'strength': strength,
                    'weight': strength * 10  # Scale for visualization
                }
                edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'theme_count': len(themes),
            'relationship_count': len(edges),
            'max_frequency': max([node['frequency'] for node in nodes]) if nodes else 1
        }
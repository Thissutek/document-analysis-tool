"""
Visualizer Module
Handles Streamlit visualization with Plotly bubble charts
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List
import numpy as np


class Visualizer:
    """Creates interactive visualizations for theme analysis"""
    
    def __init__(self):
        pass
    
    def create_bubble_chart(self, visualization_data: Dict[str, any]) -> go.Figure:
        """
        Create interactive bubble chart showing theme relationships
        
        Args:
            visualization_data: Processed data from RelationshipCalculator
            
        Returns:
            go.Figure: Plotly figure object
        """
        if not visualization_data or not visualization_data.get('nodes'):
            return self._create_empty_chart()
        
        nodes = visualization_data['nodes']
        edges = visualization_data.get('edges', [])
        
        # Create bubble chart
        fig = go.Figure()
        
        # Add theme bubbles
        x_pos = []
        y_pos = []
        sizes = []
        colors = []
        hover_text = []
        
        # Position nodes in a circular layout (simple approach)
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n_nodes
            radius = 3 + node['research_relationship'] * 2  # Closer to center = more relevant
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            x_pos.append(x)
            y_pos.append(y)
            sizes.append(max(20, node['frequency'] * 10))  # Scale bubble size
            colors.append(node['research_relationship'])
            
            # Hover text with details
            hover_text.append(
                f"<b>{node['label']}</b><br>"
                f"Description: {node['description']}<br>"
                f"Frequency: {node['frequency']}<br>"
                f"Relevance: {node['relevance_score']:.1f}/10<br>"
                f"Research Relationship: {node['research_relationship']:.2f}"
            )
        
        # Add scatter plot for bubbles
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Research<br>Relevance"),
                line=dict(width=2, color='white')
            ),
            text=[node['label'] for node in nodes],
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text,
            name="Themes"
        ))
        
        # Add relationship lines
        for edge in edges:
            # Find source and target positions
            source_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['source'])
            target_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['target'])
            
            fig.add_trace(go.Scatter(
                x=[x_pos[source_idx], x_pos[target_idx]],
                y=[y_pos[source_idx], y_pos[target_idx]],
                mode='lines',
                line=dict(
                    width=max(1, edge['strength'] * 5),
                    color='rgba(128, 128, 128, 0.5)'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Document Theme Relationships",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            plot_bgcolor='white',
            width=800,
            height=600
        )
        
        return fig
    
    def create_frequency_chart(self, themes: List[Dict]) -> go.Figure:
        """
        Create bar chart showing theme frequencies
        
        Args:
            themes: List of theme dictionaries
            
        Returns:
            go.Figure: Plotly bar chart
        """
        if not themes:
            return self._create_empty_chart("No themes to display")
        
        # Sort themes by frequency
        sorted_themes = sorted(themes, key=lambda x: x.get('chunk_frequency', 0), reverse=True)
        
        theme_names = [theme['name'] for theme in sorted_themes]
        frequencies = [theme.get('chunk_frequency', 0) for theme in sorted_themes]
        
        fig = go.Figure(data=[
            go.Bar(
                x=theme_names,
                y=frequencies,
                marker_color=px.colors.sequential.Viridis[:len(theme_names)]
            )
        ])
        
        fig.update_layout(
            title="Theme Frequency Distribution",
            xaxis_title="Themes",
            yaxis_title="Frequency (Chunk Count)",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def display_theme_summary(self, visualization_data: Dict[str, any]):
        """
        Display summary statistics and metrics using Streamlit
        
        Args:
            visualization_data: Processed visualization data
        """
        if not visualization_data:
            st.warning("No data available for summary")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Themes",
                value=visualization_data.get('theme_count', 0)
            )
        
        with col2:
            st.metric(
                label="Relationships",
                value=visualization_data.get('relationship_count', 0)
            )
        
        with col3:
            max_freq = visualization_data.get('max_frequency', 0)
            st.metric(
                label="Max Frequency",
                value=max_freq
            )
        
        with col4:
            nodes = visualization_data.get('nodes', [])
            avg_relevance = np.mean([node['research_relationship'] for node in nodes]) if nodes else 0
            st.metric(
                label="Avg Relevance",
                value=f"{avg_relevance:.2f}"
            )
    
    def display_detailed_themes(self, themes: List[Dict]):
        """
        Display detailed theme information in expandable sections
        
        Args:
            themes: List of theme dictionaries
        """
        if not themes:
            st.info("No themes extracted from the document")
            return
        
        st.subheader("Detailed Theme Analysis")
        
        for i, theme in enumerate(themes):
            with st.expander(f"🎯 {theme['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {theme.get('description', 'No description available')}")
                    st.write(f"**Relevance Score:** {theme.get('relevance_score', 'N/A')}/10")
                
                with col2:
                    st.write(f"**Frequency:** {theme.get('chunk_frequency', 0)} chunks")
                    st.write(f"**Relative Frequency:** {theme.get('relative_frequency', 0):.2%}")
                
                # Key phrases
                key_phrases = theme.get('key_phrases', [])
                if key_phrases:
                    st.write("**Key Phrases:**")
                    for phrase in key_phrases[:5]:  # Show top 5
                        st.write(f"• {phrase}")
    
    def _create_empty_chart(self, message: str = "No data available") -> go.Figure:
        """
        Create empty placeholder chart
        
        Args:
            message: Message to display
            
        Returns:
            go.Figure: Empty chart with message
        """
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16),
            xref="paper",
            yref="paper"
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        return fig
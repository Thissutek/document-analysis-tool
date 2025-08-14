"""
Chart Generators Module
Handles creation of various charts and visualizations using Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from math import pi, cos, sin
from typing import List, Dict, Any


def create_theme_confidence_chart(theme_data: List[Dict]) -> go.Figure:
    """Create theme confidence distribution histogram"""
    df_themes = pd.DataFrame(theme_data)
    
    fig = px.histogram(
        df_themes, 
        x='Confidence', 
        nbins=10,
        title="Distribution of Theme Confidence Scores",
        labels={'Confidence': 'Confidence Score (0-1)', 'count': 'Number of Themes'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(height=400)
    return fig


def create_frequency_confidence_scatter(theme_data: List[Dict]) -> go.Figure:
    """Create theme frequency vs confidence scatter plot"""
    df_themes = pd.DataFrame(theme_data)
    
    fig = px.scatter(
        df_themes,
        x='Confidence',
        y='Frequency',
        hover_name='Theme',
        color='Source',
        title="Theme Confidence vs Document Frequency",
        labels={'Confidence': 'Confidence Score (0-1)', 'Frequency': 'Document Chunks'},
        size_max=15
    )
    fig.update_layout(height=400)
    return fig


def create_alignment_chart(alignment_data: List[Dict]) -> go.Figure:
    """Create research topics vs themes alignment chart"""
    df_alignment = pd.DataFrame(alignment_data)
    
    fig = px.bar(
        df_alignment.sort_values('Alignment Score', ascending=True).tail(20),
        x='Alignment Score',
        y='Theme',
        color='Type',
        hover_data=['Research Input', 'Theme Confidence'],
        title="Top Theme-Topic Alignments",
        orientation='h',
        color_discrete_map={'Topic': '#667eea', 'Question': '#764ba2'}
    )
    fig.update_layout(height=max(400, len(df_alignment.tail(20)) * 20))
    return fig


def create_coverage_chart(topic_coverage: List[Dict]) -> go.Figure:
    """Create research topic coverage chart"""
    df_coverage = pd.DataFrame(topic_coverage)
    
    fig = px.bar(
        df_coverage,
        x='Research Input',
        y='Related Themes',
        color='Coverage Status',
        title="Research Topic Coverage Analysis",
        labels={'Related Themes': 'Number of Related Themes Found'},
        color_discrete_map={'Good': '#10b981', 'Partial': '#f59e0b', 'Poor': '#ef4444'}
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400)
    return fig


def create_interactive_chord_diagram(extracted_themes: List[Dict], relationship_analysis: Dict[str, Any]) -> go.Figure:
    """Create an interactive chord diagram showing theme relationships"""
    if not extracted_themes or len(extracted_themes) < 2:
        return go.Figure().add_annotation(
            text="Need at least 2 themes to create chord diagram",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    themes = [theme['name'] for theme in extracted_themes]
    n_themes = len(themes)
    
    # Create adjacency matrix
    matrix = [[0 for _ in range(n_themes)] for _ in range(n_themes)]
    
    if relationship_analysis and 'theme_relationships' in relationship_analysis:
        # Use relationship data if available
        relationships = relationship_analysis['theme_relationships']
        for (theme1, theme2), strength in relationships.items():
            try:
                i = themes.index(theme1)
                j = themes.index(theme2)
                matrix[i][j] = strength
                matrix[j][i] = strength
            except ValueError:
                continue
    else:
        # Fallback: calculate based on chunk overlap
        for i, theme1 in enumerate(extracted_themes):
            for j, theme2 in enumerate(extracted_themes):
                if i != j:
                    chunk_ids1 = set(theme1.get('chunk_ids', []))
                    chunk_ids2 = set(theme2.get('chunk_ids', []))
                    if chunk_ids1 and chunk_ids2:
                        overlap = len(chunk_ids1.intersection(chunk_ids2))
                        total = len(chunk_ids1.union(chunk_ids2))
                        strength = overlap / total if total > 0 else 0
                        matrix[i][j] = strength
    
    # Create chord diagram using Plotly
    # Calculate positions around circle
    angles = [2 * pi * i / n_themes for i in range(n_themes)]
    
    # Node positions
    radius = 1
    x_nodes = [radius * cos(angle) for angle in angles]
    y_nodes = [radius * sin(angle) for angle in angles]
    
    # Create traces for connections (chords)
    edge_traces = []
    for i in range(n_themes):
        for j in range(i + 1, n_themes):
            strength = matrix[i][j]
            if strength > 0.1:  # Only show meaningful connections
                # Create curved connection
                x0, y0 = x_nodes[i], y_nodes[i]
                x1, y1 = x_nodes[j], y_nodes[j]
                
                # Control points for curved line
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                control_factor = 0.3 * strength  # Curve based on strength
                ctrl_x = mid_x * (1 - control_factor)
                ctrl_y = mid_y * (1 - control_factor)
                
                # Create smooth curve with multiple points
                t_values = np.linspace(0, 1, 20)
                curve_x = []
                curve_y = []
                
                for t in t_values:
                    # Quadratic Bezier curve
                    x = (1-t)**2 * x0 + 2*(1-t)*t * ctrl_x + t**2 * x1
                    y = (1-t)**2 * y0 + 2*(1-t)*t * ctrl_y + t**2 * y1
                    curve_x.append(x)
                    curve_y.append(y)
                
                # Color and width based on strength
                if strength > 0.7:
                    color = 'rgba(0, 255, 136, 0.8)'  # Green
                    width = 8
                elif strength > 0.4:
                    color = 'rgba(255, 170, 0, 0.6)'   # Orange
                    width = 5
                else:
                    color = 'rgba(136, 170, 255, 0.4)' # Blue
                    width = 3
                
                edge_trace = go.Scatter(
                    x=curve_x, y=curve_y,
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='text',
                    text=f'{themes[i]} ↔ {themes[j]}<br>Strength: {strength:.3f}',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
    
    # Create node trace
    node_colors = []
    node_sizes = []
    hover_text = []
    
    for i, theme in enumerate(extracted_themes):
        confidence = theme.get('confidence', 0)
        frequency = theme.get('chunk_frequency', 1)
        
        # Color based on confidence
        if confidence > 0.7:
            color = '#10b981'  # Green
        elif confidence > 0.4:
            color = '#f59e0b'  # Amber
        else:
            color = '#ef4444'  # Red
            
        node_colors.append(color)
        node_sizes.append(max(20, min(60, frequency * 8 + confidence * 30)))
        
        # Count connections
        connections = sum(1 for j in range(n_themes) if matrix[i][j] > 0.1 and i != j)
        
        hover_text.append(
            f"<b>{theme['name']}</b><br>"
            f"Confidence: {confidence:.2f}<br>"
            f"Frequency: {frequency}<br>"
            f"Connections: {connections}<br>"
            f"Description: {theme.get('description', 'N/A')[:100]}..."
        )
    
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=themes,
        textposition="middle center",
        hoverinfo='text',
        hovertext=hover_text,
        textfont=dict(size=10, color='white'),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Interactive Theme Relationship Chord Diagram",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        annotations=[
            dict(
                text="Hover over nodes and connections for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color='gray')
            )
        ],
        height=600
    )
    
    return fig


def create_pyvis_network(extracted_themes: List[Dict], relationship_analysis: Dict[str, Any], 
                        layout_style: str = "Spread Out (Recommended)") -> str:
    """Create a Pyvis network graph and return HTML"""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>Pyvis not available. Install with: pip install pyvis</p>"
    
    if not extracted_themes:
        return "<p>No themes available for network visualization</p>"
    
    # Create network
    net = Network(height="500px", width="100%", bgcolor="white", font_color="black")
    
    # Layout mapping
    layout_map = {
        "Spread Out (Recommended)": "repulsion",
        "Hierarchical": "hierarchical", 
        "Circular": "circular",
        "Grid": "grid"
    }
    physics_layout = layout_map.get(layout_style, "repulsion")
    
    # Add nodes
    max_freq = max([theme.get('chunk_frequency', 1) for theme in extracted_themes]) if extracted_themes else 1
    
    for theme in extracted_themes:
        theme_name = theme['name']
        confidence = theme.get('confidence', 0)
        frequency = theme.get('chunk_frequency', 1)
        description = theme.get('description', 'No description available')
        
        # Node size based on frequency
        size = max(20, min(60, (frequency / max_freq) * 50 + 10))
        
        # Node color based on confidence
        if confidence > 0.7:
            color = '#10b981'  # Green
        elif confidence > 0.4:
            color = '#f59e0b'  # Amber  
        else:
            color = '#ef4444'  # Red
        
        # Add node with detailed info
        net.add_node(
            theme_name,
            label=theme_name,
            size=size,
            color=color,
            title=f"<b>{theme_name}</b><br>"
                  f"Confidence: {confidence:.2f}<br>"
                  f"Frequency: {frequency}<br>"
                  f"Description: {description[:150]}...",
            font={'size': max(12, min(20, size // 3))}
        )
    
    # Add edges based on relationships
    if relationship_analysis and 'theme_relationships' in relationship_analysis:
        relationships = relationship_analysis['theme_relationships']
        for (theme1, theme2), strength in relationships.items():
            if strength > 0.1:  # Only show meaningful connections
                # Edge width based on strength
                width = max(1, min(8, strength * 10))
                
                # Edge color based on strength
                if strength > 0.7:
                    color = '#10b981'  # Green
                elif strength > 0.4:
                    color = '#f59e0b'  # Amber
                else:
                    color = '#6b7280'  # Gray
                
                net.add_edge(
                    theme1, theme2,
                    width=width,
                    color=color,
                    title=f"Relationship Strength: {strength:.3f}"
                )
    else:
        # Fallback: create edges based on chunk overlap
        for i, theme1 in enumerate(extracted_themes):
            for j, theme2 in enumerate(extracted_themes[i+1:], i+1):
                chunk_ids1 = set(theme1.get('chunk_ids', []))
                chunk_ids2 = set(theme2.get('chunk_ids', []))
                
                if chunk_ids1 and chunk_ids2:
                    overlap = len(chunk_ids1.intersection(chunk_ids2))
                    total = len(chunk_ids1.union(chunk_ids2))
                    strength = overlap / total if total > 0 else 0
                    
                    if strength > 0.1:
                        width = max(1, min(8, strength * 10))
                        color = '#6b7280' if strength < 0.4 else '#f59e0b' if strength < 0.7 else '#10b981'
                        
                        net.add_edge(
                            theme1['name'], theme2['name'],
                            width=width,
                            color=color,
                            title=f"Chunk Overlap: {strength:.3f}"
                        )
    
    # Configure physics
    if physics_layout == "repulsion":
        net.repulsion(node_distance=120, central_gravity=0.33, spring_length=110, spring_strength=0.10, damping=0.95)
    elif physics_layout == "hierarchical":
        net.set_options('{"physics": {"hierarchicalRepulsion": {"nodeDistance": 120}}}')
    elif physics_layout == "circular":
        net.set_options('{"physics": {"enabled": false}, "layout": {"randomSeed": 2}}')
    elif physics_layout == "grid":
        net.set_options('{"physics": {"enabled": false}}')
    
    # Generate HTML
    html = net.generate_html()
    
    return html


def create_network_visualization(extracted_themes: List[Dict], relationship_analysis: Dict[str, Any]) -> go.Figure:
    """Create a network visualization using Plotly"""
    if not extracted_themes or len(extracted_themes) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 themes to create network visualization",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Prepare node data
    theme_names = [theme['name'] for theme in extracted_themes]
    n_themes = len(theme_names)
    
    # Simple circular layout for nodes
    angles = [2 * pi * i / n_themes for i in range(n_themes)]
    x_nodes = [cos(angle) for angle in angles]
    y_nodes = [sin(angle) for angle in angles]
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_info = []
    
    if relationship_analysis and 'theme_relationships' in relationship_analysis:
        relationships = relationship_analysis['theme_relationships']
        for (theme1, theme2), strength in relationships.items():
            if strength > 0.2:  # Only show strong connections
                try:
                    i1 = theme_names.index(theme1)
                    i2 = theme_names.index(theme2)
                    
                    edge_x.extend([x_nodes[i1], x_nodes[i2], None])
                    edge_y.extend([y_nodes[i1], y_nodes[i2], None])
                    edge_info.append(f"{theme1} ↔ {theme2}: {strength:.3f}")
                except ValueError:
                    continue
    else:
        # Fallback: calculate relationships from chunk overlap
        for i, theme1 in enumerate(extracted_themes):
            for j, theme2 in enumerate(extracted_themes[i+1:], i+1):
                chunk_ids1 = set(theme1.get('chunk_ids', []))
                chunk_ids2 = set(theme2.get('chunk_ids', []))
                
                if chunk_ids1 and chunk_ids2:
                    overlap = len(chunk_ids1.intersection(chunk_ids2))
                    total = len(chunk_ids1.union(chunk_ids2))
                    strength = overlap / total if total > 0 else 0
                    
                    if strength > 0.2:
                        edge_x.extend([x_nodes[i], x_nodes[j], None])
                        edge_y.extend([y_nodes[i], y_nodes[j], None])
                        edge_info.append(f"{theme1['name']} ↔ {theme2['name']}: {strength:.3f}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightgray'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_sizes = []
    node_colors = []
    hover_text = []
    
    for theme in extracted_themes:
        confidence = theme.get('confidence', 0)
        frequency = theme.get('chunk_frequency', 1)
        
        # Size based on frequency
        size = max(20, min(60, frequency * 5 + 10))
        node_sizes.append(size)
        
        # Color based on confidence
        if confidence > 0.7:
            color = '#10b981'  # Green
        elif confidence > 0.4:
            color = '#f59e0b'  # Amber
        else:
            color = '#ef4444'  # Red
        node_colors.append(color)
        
        # Hover text
        hover_text.append(
            f"<b>{theme['name']}</b><br>"
            f"Confidence: {confidence:.2f}<br>"
            f"Frequency: {frequency}<br>"
            f"Description: {theme.get('description', 'N/A')[:100]}..."
        )
    
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=theme_names,
        textposition="middle center",
        hoverinfo='text',
        hovertext=hover_text,
        textfont=dict(size=10, color='white')
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Theme Network Visualization",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        annotations=[
            dict(
                text="Node size = frequency, Color = confidence level",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color='gray')
            )
        ],
        height=600
    )
    
    return fig

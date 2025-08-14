import streamlit as st
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Bootstrap icons and modern UI helper functions
def load_bootstrap_css():
    """Load Bootstrap Icons and custom CSS for modern UI"""
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <style>
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .theme-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .chunk-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .status-success { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-error { color: #ef4444; }
    .icon { margin-right: 8px; }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon, description=None):
    """Create a modern metric card"""
    desc_html = f'<p style="margin: 0; color: #6b7280; font-size: 0.9rem;">{description}</p>' if description else ''
    return f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <i class="bi bi-{icon} icon" style="font-size: 1.2rem; color: #667eea;"></i>
            <span style="font-weight: 600; color: #374151;">{title}</span>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #111827; margin-bottom: 0.25rem;">{value}</div>
        {desc_html}
    </div>
    """

def get_bootstrap_icon(icon_name):
    """Return Bootstrap icon HTML"""
    return f'<i class="bi bi-{icon_name} icon"></i>'

def create_research_analysis_card(research_topics, topic_warnings):
    """Create a card showing the AI's understanding of research topics"""
    topics_html = ""
    if research_topics:
        topics_html = "<ul style='margin: 0; padding-left: 20px;'>"
        for topic in research_topics[:8]:  # Show first 8 topics
            topics_html += f"<li style='margin-bottom: 4px; color: #374151;'>{topic}</li>"
        if len(research_topics) > 8:
            topics_html += f"<li style='color: #6b7280; font-style: italic;'>...and {len(research_topics) - 8} more</li>"
        topics_html += "</ul>"
    
    warnings_html = ""
    if topic_warnings:
        warnings_html = f"""
        <div style="margin-top: 12px; padding: 8px; background: #fef3cd; border-radius: 6px; border-left: 3px solid #f59e0b;">
            <div style="font-size: 0.9rem; color: #92400e; font-weight: 500;">
                <i class="bi bi-exclamation-triangle" style="margin-right: 4px;"></i>
                Processing Notes:
            </div>
            <div style="font-size: 0.8rem; color: #92400e; margin-top: 4px;">
                {len(topic_warnings)} topics were adjusted during processing
            </div>
        </div>
        """
    
    return f"""
    <div class="metric-card" style="margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <i class="bi bi-lightbulb icon" style="font-size: 1.2rem; color: #667eea;"></i>
            <span style="font-weight: 600; color: #374151; font-size: 1.1rem;">AI Research Understanding</span>
        </div>
        <div style="color: #6b7280; margin-bottom: 12px; font-size: 0.95rem;">
            The AI processed your input and identified these key research areas:
        </div>
        {topics_html}
        <div style="margin-top: 12px; font-size: 0.9rem; color: #6b7280;">
            <strong>{len(research_topics)}</strong> research topics • Analysis will focus on finding themes related to these areas
        </div>
        {warnings_html}
    </div>
    """


def validate_and_format_topics(topics_list):
    """
    Validate and format research topics
    
    Args:
        topics_list: List of topic strings
        
    Returns:
        tuple: (valid_topics, warnings)
    """
    valid_topics = []
    warnings = []
    
    for topic in topics_list:
        # Clean up topic
        cleaned_topic = topic.strip()
        
        # Skip empty topics
        if not cleaned_topic:
            continue
            
        # Check minimum length
        if len(cleaned_topic) < 3:
            warnings.append(f"Topic too short (skipped): '{cleaned_topic}'")
            continue
            
        # Check maximum length
        if len(cleaned_topic) > 200:
            warnings.append(f"Topic too long (truncated): '{cleaned_topic[:50]}...'")
            cleaned_topic = cleaned_topic[:200]
            
        # Remove excessive punctuation and normalize
        cleaned_topic = re.sub(r'[^\w\s\-\?\!\.]+', '', cleaned_topic)
        cleaned_topic = re.sub(r'\s+', ' ', cleaned_topic)  # Remove extra spaces
        
        # Capitalize first letter
        cleaned_topic = cleaned_topic[0].upper() + cleaned_topic[1:] if len(cleaned_topic) > 1 else cleaned_topic.upper()
        
        # Check for duplicates (case insensitive)
        if cleaned_topic.lower() not in [t.lower() for t in valid_topics]:
            valid_topics.append(cleaned_topic)
        else:
            warnings.append(f"Duplicate topic removed: '{cleaned_topic}'")
    
    return valid_topics, warnings

def create_theme_confidence_chart(theme_data):
    """Create theme confidence distribution histogram"""
    import plotly.express as px
    import pandas as pd
    
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

def create_frequency_confidence_scatter(theme_data):
    """Create theme frequency vs confidence scatter plot"""
    import plotly.express as px
    import pandas as pd
    
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

def create_alignment_chart(alignment_data):
    """Create research topics vs themes alignment chart"""
    import plotly.express as px
    import pandas as pd
    
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

def create_coverage_chart(topic_coverage):
    """Create research topic coverage chart"""
    import plotly.express as px
    import pandas as pd
    
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

def create_pyvis_network(extracted_themes, relationship_analysis):
    """Create interactive PyVis network visualization"""
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
        import tempfile
        import os
        
        # Prepare themes data
        themes = []
        for theme in extracted_themes:
            confidence = theme.get('confidence', 0)
            frequency = theme.get('chunk_frequency', 1)
            
            # Determine cluster based on confidence
            if confidence > 0.7:
                cluster = "High Confidence"
                color = "#10b981"
            elif confidence > 0.4:
                cluster = "Medium Confidence"
                color = "#f59e0b"
            else:
                cluster = "Low Confidence"
                color = "#ef4444"
            
            # Size based on frequency and confidence
            size = max(15, min(60, frequency * 10 + confidence * 30))
            
            themes.append({
                "id": theme['name'],
                "size": size,
                "cluster": cluster,
                "color": color,
                "confidence": confidence,
                "frequency": frequency,
                "description": theme.get('description', 'No description available')
            })
        
        # Prepare relationships data
        relationships = []
        if relationship_analysis and 'theme_relationships' in relationship_analysis:
            for rel in relationship_analysis['theme_relationships']:
                source = rel.get('theme1', '')
                target = rel.get('theme2', '')
                weight = rel.get('strength', 0.1)
                
                if weight > 0.1:  # Only meaningful relationships
                    relationships.append({
                        "source": source,
                        "target": target,
                        "weight": weight
                    })
        
        # If no explicit relationships, create based on co-occurrence
        if not relationships:
            for i, theme1 in enumerate(extracted_themes):
                for j, theme2 in enumerate(extracted_themes[i+1:], i+1):
                    # Safely get chunk_ids and convert to set
                    chunk_ids1 = theme1.get('chunk_ids', [])
                    chunk_ids2 = theme2.get('chunk_ids', [])
                    
                    # Ensure we have lists/iterables, not sets
                    if hasattr(chunk_ids1, '__iter__') and not isinstance(chunk_ids1, str):
                        theme1_chunks = set(chunk_ids1) if chunk_ids1 else set()
                    else:
                        theme1_chunks = set()
                    
                    if hasattr(chunk_ids2, '__iter__') and not isinstance(chunk_ids2, str):
                        theme2_chunks = set(chunk_ids2) if chunk_ids2 else set()
                    else:
                        theme2_chunks = set()
                    
                    if theme1_chunks and theme2_chunks:
                        overlap = len(theme1_chunks.intersection(theme2_chunks))
                        total = len(theme1_chunks.union(theme2_chunks))
                        weight = overlap / total if total > 0 else 0
                        
                        if weight > 0.15:  # Meaningful co-occurrence
                            relationships.append({
                                "source": theme1['name'],
                                "target": theme2['name'],
                                "weight": weight
                            })
        
        # Create PyVis network with modern dark theme
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#1a1a1a",
            font_color="white",
            directed=False
        )
        
        # Use Barnes-Hut physics for better performance and layout
        net.barnes_hut()
        
        # Configure modern physics and styling
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 150},
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.1,
              "springLength": 120,
              "springConstant": 0.02,
              "damping": 0.15,
              "avoidOverlap": 0.2
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "selectConnectedEdges": false,
            "hoverConnectedEdges": true
          },
          "nodes": {
            "font": {
              "size": 16,
              "color": "white",
              "face": "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
            },
            "borderWidth": 3,
            "borderWidthSelected": 5,
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.6)",
              "size": 15,
              "x": 3,
              "y": 3
            },
            "scaling": {
              "min": 15,
              "max": 60
            },
            "chosen": {
              "node": "function(values, id, selected, hovering) { values.shadow = true; values.shadowSize = 20; }"
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous",
              "roundness": 0.5
            },
            "shadow": {
              "enabled": true,
              "color": "rgba(255,255,255,0.1)",
              "size": 8,
              "x": 2,
              "y": 2
            },
            "color": {
              "inherit": false,
              "opacity": 0.7
            },
            "chosen": {
              "edge": "function(values, id, selected, hovering) { values.opacity = 1; }"
            }
          }
        }
        """)
        
        # First pass: Add all nodes
        theme_id_map = {}
        for i, theme in enumerate(themes):
            theme_id = theme['id']
            theme_id_map[theme_id] = i
            
            net.add_node(
                theme_id,
                label=theme_id[:25] + '...' if len(theme_id) > 25 else theme_id,
                title=theme_id,  # Will be enhanced later with neighbors
                size=theme['size'],
                color=theme['color'],
                group=theme['cluster'],
                value=theme['frequency'],  # Used for size scaling
                confidence=theme['confidence'],
                description=theme['description']
            )
        
        # Add edges with modern styling
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            
            if source in theme_id_map and target in theme_id_map:
                # Edge width and color based on weight
                edge_width = max(1, rel['weight'] * 12)
                
                # Modern gradient colors based on relationship strength
                if rel['weight'] > 0.7:
                    edge_color = "#00d4aa"  # Strong - teal
                elif rel['weight'] > 0.4:
                    edge_color = "#fbbf24"  # Medium - amber  
                else:
                    edge_color = "#64748b"  # Weak - slate
                
                net.add_edge(
                    source,
                    target,
                    value=rel['weight'],  # This affects edge thickness
                    color=edge_color,
                    title=f"Relationship Strength: {rel['weight']:.3f}",
                    width=edge_width
                )
        
        # Add neighbor information to tooltips (like Game of Thrones example)
        try:
            neighbor_map = net.get_adj_list()
        except Exception as e:
            print(f"Error getting adjacency list: {e}")
            neighbor_map = {}
        
        # Enhanced tooltips with neighbor data and topic relations
        for node in net.nodes:
            try:
                node_id = node.get("id", "Unknown")
                neighbors = neighbor_map.get(node_id, [])
                
                # Get original theme data
                theme_info = next((t for t in themes if t.get("id") == node_id), {})
                
                # Ensure we have valid data
                confidence = theme_info.get('confidence', 0)
                frequency = theme_info.get('frequency', 0)
                cluster = theme_info.get('cluster', 'Unknown')
                description = theme_info.get('description', 'No description available')
                
                # Create rich tooltip with modern dark styling
                tooltip_html = f"""
                <div style='
                    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                    color: white;
                    padding: 16px;
                    border-radius: 12px;
                    max-width: 350px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    font-family: Inter, -apple-system, sans-serif;
                '>
                    <h3 style='margin: 0 0 12px 0; color: #f1f5f9; font-size: 18px; font-weight: 600;'>
                        {node_id}
                    </h3>
                    
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;'>
                        <div style='background: rgba(16, 185, 129, 0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #10b981;'>
                            <div style='font-size: 12px; color: #94a3b8;'>Confidence</div>
                            <div style='font-size: 16px; font-weight: 600; color: #10b981;'>{confidence:.2f}</div>
                        </div>
                        <div style='background: rgba(59, 130, 246, 0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #3b82f6;'>
                            <div style='font-size: 12px; color: #94a3b8;'>Frequency</div>
                            <div style='font-size: 16px; font-weight: 600; color: #3b82f6;'>{frequency}</div>
                        </div>
                    </div>
                    
                    <div style='margin-bottom: 12px;'>
                        <div style='font-size: 14px; color: #cbd5e1; margin-bottom: 4px;'>Cluster: {cluster}</div>
                        <div style='font-size: 13px; color: #94a3b8; line-height: 1.4;'>
                            {description[:120]}{'...' if len(description) > 120 else ''}
                        </div>
                    </div>
                """
            
                if neighbors:
                    # Add related themes section
                    tooltip_html += f"""
                    <div style='border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 12px;'>
                        <div style='font-size: 14px; font-weight: 600; color: #f1f5f9; margin-bottom: 8px;'>
                            🔗 Related Themes ({len(neighbors)}):
                        </div>
                        <div style='display: flex; flex-wrap: wrap; gap: 6px;'>
                    """
                    
                    # Show up to 5 neighbors as tags
                    for neighbor in neighbors[:5]:
                        neighbor_theme = next((t for t in themes if t.get("id") == neighbor), {})
                        neighbor_confidence = neighbor_theme.get('confidence', 0)
                        
                        # Color based on confidence
                        if neighbor_confidence > 0.7:
                            tag_color = "#10b981"
                        elif neighbor_confidence > 0.4:
                            tag_color = "#f59e0b" 
                        else:
                            tag_color = "#64748b"
                        
                        tooltip_html += f"""
                            <span style='
                                background: {tag_color}20;
                                color: {tag_color};
                                padding: 4px 8px;
                                border-radius: 16px;
                                font-size: 12px;
                                border: 1px solid {tag_color}40;
                            '>
                                {str(neighbor)[:20]}{'...' if len(str(neighbor)) > 20 else ''}
                            </span>
                        """
                    
                    if len(neighbors) > 5:
                        tooltip_html += f"""
                            <span style='
                                background: rgba(148, 163, 184, 0.1);
                                color: #94a3b8;
                                padding: 4px 8px;
                                border-radius: 16px;
                                font-size: 12px;
                                border: 1px solid rgba(148, 163, 184, 0.2);
                            '>
                                +{len(neighbors) - 5} more
                            </span>
                        """
                    
                    tooltip_html += "</div></div>"
                
                tooltip_html += "</div>"
                
                # Update node with enhanced tooltip and neighbor count
                node["title"] = tooltip_html
                node["value"] = len(neighbors)  # This affects node size
                
            except Exception as e:
                print(f"Error processing node {node.get('id', 'unknown')}: {e}")
                # Fallback to simple tooltip
                node["title"] = f"Theme: {node.get('id', 'Unknown')}"
                node["value"] = 1
        
        # Generate and return HTML
        if len(themes) > 0:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                net.save_graph(tmp.name)
                tmp_path = tmp.name
            
            # Read the HTML content
            with open(tmp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return html_content
        else:
            return None
            
    except ImportError:
        return "PyVis not available - install with: pip install pyvis"
    except Exception as e:
        return f"Error creating PyVis network: {str(e)}"

def create_network_visualization(extracted_themes, relationship_analysis):
    """Create interactive network visualization of themes and their relationships"""
    try:
        import plotly.graph_objects as go
        import networkx as nx
        import pandas as pd
        import numpy as np
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (themes)
        node_data = []
        for i, theme in enumerate(extracted_themes):
            theme_name = theme['name']
            confidence = theme.get('confidence', 0)
            frequency = theme.get('chunk_frequency', 1)
            
            # Determine cluster/color based on confidence
            if confidence > 0.7:
                cluster = 'High Confidence'
                color = '#10b981'
            elif confidence > 0.4:
                cluster = 'Medium Confidence' 
                color = '#f59e0b'
            else:
                cluster = 'Low Confidence'
                color = '#ef4444'
            
            # Node size based on frequency and confidence
            size = max(10, min(50, frequency * 15 + confidence * 20))
            
            G.add_node(theme_name, 
                      confidence=confidence,
                      frequency=frequency,
                      cluster=cluster,
                      color=color,
                      size=size)
            
            node_data.append({
                'name': theme_name,
                'confidence': confidence,
                'frequency': frequency,
                'cluster': cluster,
                'color': color,
                'size': size
            })
        
        # Add edges (relationships)
        edge_data = []
        if relationship_analysis and 'theme_relationships' in relationship_analysis:
            relationships = relationship_analysis['theme_relationships']
            
            for rel in relationships:
                source = rel.get('theme1', '')
                target = rel.get('theme2', '')
                weight = rel.get('strength', 0.1)
                
                if source in G.nodes and target in G.nodes and weight > 0.1:
                    G.add_edge(source, target, weight=weight)
                    edge_data.append({
                        'source': source,
                        'target': target,
                        'weight': weight
                    })
        
        # If no relationships, create connections based on co-occurrence in chunks
        if not edge_data:
            # Create artificial relationships based on themes appearing in same chunks
            for i, theme1 in enumerate(extracted_themes):
                for j, theme2 in enumerate(extracted_themes[i+1:], i+1):
                    theme1_chunks = set(theme1.get('chunk_ids', []))
                    theme2_chunks = set(theme2.get('chunk_ids', []))
                    
                    # Calculate co-occurrence
                    if theme1_chunks and theme2_chunks:
                        overlap = len(theme1_chunks.intersection(theme2_chunks))
                        total = len(theme1_chunks.union(theme2_chunks))
                        weight = overlap / total if total > 0 else 0
                        
                        if weight > 0.1:  # Only add meaningful connections
                            G.add_edge(theme1['name'], theme2['name'], weight=weight)
                            edge_data.append({
                                'source': theme1['name'],
                                'target': theme2['name'],
                                'weight': weight
                            })
        
        # Generate layout
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Extract node positions and data
            node_trace_high = go.Scatter(x=[], y=[], mode='markers+text', 
                                        name='High Confidence', 
                                        marker=dict(color='#10b981', size=[], line=dict(width=2)),
                                        text=[], textposition="middle center",
                                        hovertemplate='<b>%{text}</b><br>Confidence: %{customdata[0]:.2f}<br>Frequency: %{customdata[1]}<extra></extra>',
                                        customdata=[])
            
            node_trace_med = go.Scatter(x=[], y=[], mode='markers+text',
                                       name='Medium Confidence',
                                       marker=dict(color='#f59e0b', size=[], line=dict(width=2)),
                                       text=[], textposition="middle center",
                                       hovertemplate='<b>%{text}</b><br>Confidence: %{customdata[0]:.2f}<br>Frequency: %{customdata[1]}<extra></extra>',
                                       customdata=[])
            
            node_trace_low = go.Scatter(x=[], y=[], mode='markers+text',
                                       name='Low Confidence', 
                                       marker=dict(color='#ef4444', size=[], line=dict(width=2)),
                                       text=[], textposition="middle center",
                                       hovertemplate='<b>%{text}</b><br>Confidence: %{customdata[0]:.2f}<br>Frequency: %{customdata[1]}<extra></extra>',
                                       customdata=[])
            
            # Add nodes to appropriate traces
            for node in G.nodes():
                x, y = pos[node]
                node_info = G.nodes[node]
                confidence = node_info['confidence']
                frequency = node_info['frequency']
                size = node_info['size']
                
                # Truncate long theme names for display
                display_name = node[:15] + '...' if len(node) > 15 else node
                
                if confidence > 0.7:
                    node_trace_high['x'] += tuple([x])
                    node_trace_high['y'] += tuple([y])
                    node_trace_high['text'] += tuple([display_name])
                    node_trace_high['marker']['size'] += tuple([size])
                    node_trace_high['customdata'] += tuple([[confidence, frequency]])
                elif confidence > 0.4:
                    node_trace_med['x'] += tuple([x])
                    node_trace_med['y'] += tuple([y])
                    node_trace_med['text'] += tuple([display_name])
                    node_trace_med['marker']['size'] += tuple([size])
                    node_trace_med['customdata'] += tuple([[confidence, frequency]])
                else:
                    node_trace_low['x'] += tuple([x])
                    node_trace_low['y'] += tuple([y])
                    node_trace_low['text'] += tuple([display_name])
                    node_trace_low['marker']['size'] += tuple([size])
                    node_trace_low['customdata'] += tuple([[confidence, frequency]])
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G.edges[edge]['weight']
                
                edge_trace = go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                       mode='lines',
                                       line=dict(width=max(1, weight * 10), color='rgba(125, 125, 125, 0.3)'),
                                       hoverinfo='none',
                                       showlegend=False)
                edge_traces.append(edge_trace)
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace_high, node_trace_med, node_trace_low])
            
            fig.update_layout(
                title="Theme Relationship Network",
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Node size = frequency + confidence | Edge thickness = relationship strength",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12, color="gray")
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            return fig
        else:
            return None
            
    except Exception as e:
        print(f"Error creating network visualization: {e}")
        return None

def calculate_alignment_data(research_topics, extracted_themes):
    """Calculate topic-theme alignment data"""
    alignment_data = []
    for topic in research_topics:
        topic_lower = topic.lower()
        topic_type = "Question" if "?" in topic else "Topic"
        
        for theme in extracted_themes:
            theme_name = theme['name']
            theme_lower = theme_name.lower()
            
            # Calculate alignment
            topic_words = set(topic_lower.split())
            theme_words = set(theme_lower.split())
            
            if topic_words and theme_words:
                word_alignment = len(topic_words.intersection(theme_words)) / len(topic_words.union(theme_words))
            else:
                word_alignment = 0
            
            # Check description alignment
            description = theme.get('description', '').lower()
            desc_alignment = sum(1 for word in topic_words if word in description) / len(topic_words) if topic_words else 0
            
            final_alignment = max(word_alignment, desc_alignment * 0.8)
            
            if final_alignment > 0.1:  # Only show meaningful alignments
                alignment_data.append({
                    'Research Input': topic[:50] + '...' if len(topic) > 50 else topic,
                    'Type': topic_type,
                    'Theme': theme_name,
                    'Alignment Score': final_alignment,
                    'Theme Confidence': theme.get('confidence', 0),
                    'Full Topic': topic
                })
    
    return alignment_data

def calculate_topic_coverage(research_topics, extracted_themes):
    """Calculate topic coverage statistics"""
    topic_coverage = []
    for topic in research_topics:
        topic_lower = topic.lower()
        topic_type = "Question" if "?" in topic else "Topic"
        related_count = 0
        max_alignment = 0
        
        for theme in extracted_themes:
            theme_lower = theme['name'].lower()
            topic_words = set(topic_lower.split())
            theme_words = set(theme_lower.split())
            
            if topic_words and theme_words:
                alignment = len(topic_words.intersection(theme_words)) / len(topic_words.union(theme_words))
            else:
                alignment = 0
            
            description = theme.get('description', '').lower()
            desc_alignment = sum(1 for word in topic_words if word in description) / len(topic_words) if topic_words else 0
            final_alignment = max(alignment, desc_alignment * 0.8)
            
            if final_alignment > 0.15:
                related_count += 1
                max_alignment = max(max_alignment, final_alignment)
        
        topic_coverage.append({
            'Research Input': topic[:40] + '...' if len(topic) > 40 else topic,
            'Type': topic_type,
            'Related Themes': related_count,
            'Best Alignment': max_alignment,
            'Coverage Status': 'Good' if related_count >= 2 else 'Partial' if related_count == 1 else 'Poor'
        })
    
    return topic_coverage

def generate_insights(topic_coverage, theme_data, alignment_data):
    """Generate AI insights based on analysis results"""
    insights = []
    
    # Coverage insights
    good_coverage = len([t for t in topic_coverage if t['Coverage Status'] == 'Good'])
    poor_coverage = len([t for t in topic_coverage if t['Coverage Status'] == 'Poor'])
    
    if good_coverage > poor_coverage:
        insights.append("✅ Most research topics have good theme coverage in the document")
    elif poor_coverage > good_coverage:
        insights.append("⚠️ Many research topics lack related themes - consider refining topics or checking document relevance")
    else:
        insights.append("📊 Mixed coverage - some topics well represented, others not found")
    
    # Confidence insights
    high_conf_themes = len([t for t in theme_data if t['Confidence'] > 0.7])
    total_themes = len(theme_data)
    
    if total_themes > 0:
        if high_conf_themes / total_themes > 0.6:
            insights.append("🎯 High-quality analysis with most themes having strong confidence scores")
        elif high_conf_themes / total_themes > 0.3:
            insights.append("📈 Moderate analysis quality - consider adjusting relevance threshold")
        else:
            insights.append("⚡ Lower confidence themes - document may not strongly match research topics")
    
    # Alignment insights
    if alignment_data:
        import pandas as pd
        df_alignment = pd.DataFrame(alignment_data)
        avg_alignment = df_alignment['Alignment Score'].mean()
        
        if avg_alignment > 0.4:
            insights.append("🔗 Strong alignment between research topics and extracted themes")
        elif avg_alignment > 0.2:
            insights.append("🔍 Moderate alignment - some thematic overlap found")
        else:
            insights.append("❓ Weak alignment - extracted themes may be broader than research focus")
    
    return insights

def _display_theme_relevance_charts(extracted_themes, research_topics, viz_data):
    """Display interactive charts showing theme relevance and relationships"""
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    if not extracted_themes:
        st.warning("No themes available for visualization")
        return
    
    # Chart 1: Theme Confidence vs Frequency
    st.write("**Theme Confidence and Frequency Analysis:**")
    
    theme_data = []
    for theme in extracted_themes:
        # Get importance from viz_data nodes
        importance = 0.5  # default
        for node in viz_data.get('nodes', []):
            if node.get('id') == theme['name']:
                importance = node.get('importance', 0.5)
                break
        
        theme_data.append({
            'Theme': theme['name'],
            'Confidence': theme.get('confidence', 0),
            'Frequency': theme.get('chunk_frequency', 0),
            'Source': theme.get('source', 'unknown'),
            'Importance': importance
        })
    
    df = pd.DataFrame(theme_data)
    
    if not df.empty:
        # Scatter plot: Confidence vs Frequency
        fig1 = px.scatter(
            df, 
            x='Confidence', 
            y='Frequency',
            size='Importance',
            color='Source',
            hover_name='Theme',
            title="Theme Confidence vs Document Frequency",
            labels={'Confidence': 'Confidence Score (0-1)', 'Frequency': 'Document Chunks'},
            size_max=20
        )
        
        fig1.update_layout(
            xaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Research Topics vs Extracted Themes Alignment
    st.write("**Research Topics vs Extracted Themes Alignment:**")
    
    # Calculate topic-theme alignment scores
    topic_theme_data = []
    for topic in research_topics:
        topic_lower = topic.lower()
        for theme in extracted_themes:
            theme_name = theme['name']
            theme_lower = theme_name.lower()
            
            # Simple alignment calculation based on word overlap
            topic_words = set(topic_lower.split())
            theme_words = set(theme_lower.split())
            
            if topic_words and theme_words:
                alignment = len(topic_words.intersection(theme_words)) / len(topic_words.union(theme_words))
            else:
                alignment = 0
            
            # Also check if topic words appear in theme description
            description = theme.get('description', '').lower()
            description_alignment = sum(1 for word in topic_words if word in description) / len(topic_words) if topic_words else 0
            
            final_alignment = max(alignment, description_alignment * 0.8)  # Weight description less
            
            if final_alignment > 0.1:  # Only show meaningful alignments
                topic_theme_data.append({
                    'Research Topic': topic,
                    'Extracted Theme': theme_name,
                    'Alignment Score': final_alignment,
                    'Theme Confidence': theme.get('confidence', 0),
                    'Theme Frequency': theme.get('chunk_frequency', 0)
                })
    
    if topic_theme_data:
        alignment_df = pd.DataFrame(topic_theme_data)
        
        # Bar chart showing alignment scores
        fig3 = px.bar(
            alignment_df.sort_values('Alignment Score', ascending=True),
            x='Alignment Score',
            y='Extracted Theme',
            color='Research Topic',
            title="Theme Alignment with Research Topics",
            orientation='h',
            hover_data=['Theme Confidence', 'Theme Frequency']
        )
        
        fig3.update_layout(height=max(300, len(topic_theme_data) * 25))
        st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("No strong alignments found between research topics and extracted themes")
    
    # Chart 3: Theme Relationship Network (if relationships exist)
    edges = viz_data.get('edges', [])
    if edges:
        st.write("**Theme Relationship Strengths:**")
        
        # Create network visualization data
        edge_data = []
        for edge in edges:
            edge_data.append({
                'Source Theme': edge['source'],
                'Target Theme': edge['target'],
                'Relationship Strength': edge['strength'],
                'Co-occurrence': edge.get('cooccurrence_count', 0)
            })
        
        if edge_data:
            edge_df = pd.DataFrame(edge_data)
            
            # Bar chart of relationship strengths
            fig4 = px.bar(
                edge_df.sort_values('Relationship Strength', ascending=True),
                x='Relationship Strength',
                y=[f"{row['Source Theme']} ↔ {row['Target Theme']}" for _, row in edge_df.iterrows()],
                title="Theme Relationship Strengths",
                orientation='h',
                hover_data=['Co-occurrence']
            )
            
            fig4.update_layout(height=max(300, len(edge_data) * 30))
            st.plotly_chart(fig4, use_container_width=True)
    
    # Summary metrics
    if extracted_themes:
        st.write("**Visualization Summary:**")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            avg_confidence = sum(theme.get('confidence', 0) for theme in extracted_themes) / len(extracted_themes)
            st.metric("Average Theme Confidence", f"{avg_confidence:.2f}")
        
        with summary_col2:
            total_frequency = sum(theme.get('chunk_frequency', 0) for theme in extracted_themes)
            st.metric("Total Theme Frequency", total_frequency)
        
        with summary_col3:
            if topic_theme_data:
                avg_alignment = sum(item['Alignment Score'] for item in topic_theme_data) / len(topic_theme_data)
                st.metric("Average Topic Alignment", f"{avg_alignment:.2f}")
            else:
                st.metric("Average Topic Alignment", "N/A")

# Import custom modules (will be created)
try:
    from src.document_parser import DocumentParser
    from src.text_chunker import TextChunker
    from src.theme_analyzer import ThemeAnalyzer
    from src.relationship_calc import RelationshipCalculator
    from src.visualizer import Visualizer
except ImportError:
    st.error("Source modules not found. Please ensure all modules in src/ are properly created.")

def main():
    st.set_page_config(
        page_title="Document Theme Analysis Tool",
        layout="wide"
    )
    
    # Load Bootstrap CSS and modern styling
    load_bootstrap_css()
    
    st.title("Document Theme Analysis Tool")
    st.markdown("Analyze large text documents to identify themes and visualize their relationships")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("OpenAI API key not found. Please set your OPENAI_API_KEY in the .env file.")
        st.info("1. Copy .env.template to .env\n2. Add your OpenAI API key\n3. Restart the application")
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Research Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX document for theme analysis"
        )
        
        st.divider()
        
        # Research topics/themes input section
        st.subheader("Research Focus")
        st.markdown("*Specify the topics or themes you want to search for in the document*")
        
        # Custom topics input
        st.write("**Custom Topics:**")
        custom_topics_text = st.text_area(
            "Enter your specific topics (one per line)",
            placeholder="Example:\nEmployee motivation\nWorkplace productivity\nTeam collaboration\nRemote work challenges",            height=120,
            help="Enter each topic or theme on a separate line"
        )
        
        # Research questions
        st.write("**Research Questions:**")
        research_questions = st.text_area(
            "What specific questions do you want answered?",
            placeholder="Example:\nHow does leadership style affect team performance?\nWhat factors contribute to employee satisfaction?\nWhat are the main barriers to innovation?",            help="Enter specific questions you want the analysis to address"
        )
        
        # Combine all inputs
        raw_topics = []
        
        # Add custom topics
        if custom_topics_text.strip():
            custom_list = [topic.strip() for topic in custom_topics_text.split('\n') if topic.strip()]
            raw_topics.extend(custom_list)
        
        # Add research questions as topics
        if research_questions.strip():
            questions_list = [q.strip() for q in research_questions.split('\n') if q.strip()]
            raw_topics.extend(questions_list)
        
        # Validate and format topics
        all_topics, topic_warnings = validate_and_format_topics(raw_topics)
        
        # Display warnings if any
        if topic_warnings:
            with st.expander("Topic Processing Warnings", expanded=False):
                for warning in topic_warnings:
                    st.warning(warning)
        
        # Display current topics
        if all_topics:
            st.write("**Selected Topics/Themes:**")
            
            # Show topics in a more organized way
            if len(all_topics) <= 5:
                for i, topic in enumerate(all_topics, 1):
                    st.write(f"{i}. {topic}")
            else:
                # For many topics, show in columns
                topic_display_cols = st.columns(2)
                for i, topic in enumerate(all_topics):
                    with topic_display_cols[i % 2]:
                        st.write(f"{i+1}. {topic}")
            
            # Show topic count
            if len(all_topics) > 10:
                st.info(f"{len(all_topics)} topics selected. Consider focusing on fewer topics for better analysis quality.")
            
            # Analysis settings
            st.divider()
            st.subheader("Analysis Settings")
            
            similarity_threshold = st.slider(
                "Theme Relevance Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="How closely content must match your topics (higher = more strict)"
            )
            
            max_themes = st.slider(
                "Maximum Themes to Extract",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="Maximum number of themes to identify in the document"
            )
        
        st.divider()
        
        # Analysis button
        can_analyze = uploaded_file is not None and len(all_topics) > 0
        
        if can_analyze:
            analyze_button = st.button("Start Analysis", type="primary", use_container_width=True)
        else:
            st.button("Start Analysis", disabled=True, use_container_width=True)
            if uploaded_file is None:
                st.caption("Upload a document first")
            if len(all_topics) == 0:
                st.caption("Select or enter topics to search for")
    
    # Main content area
    if can_analyze and 'analyze_button' in locals() and analyze_button:
        # Create a single progress bar and status display
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # Initialize progress tracking
            total_steps = 6
            current_step = 0
            
            # Step 1: Initialize components
            current_step += 1
            progress_placeholder.progress(current_step / total_steps)
            status_placeholder.info(f"Step {current_step}/{total_steps}: Initializing analysis components...")
            
            parser = DocumentParser()
            chunker = TextChunker(chunk_tokens=1000, overlap_tokens=100)
            analyzer = ThemeAnalyzer()
            calc = RelationshipCalculator()
            
            # Step 2: Extract and process document
            current_step += 1
            progress_placeholder.progress(current_step / total_steps)
            status_placeholder.info(f"Step {current_step}/{total_steps}: Extracting text from document...")
            
            extracted_text = parser.extract_text_from_document(uploaded_file)
            if not extracted_text:
                st.error("Failed to extract text from document")
                st.stop()
            
            chunks = chunker.chunk_text(extracted_text)
            if not chunks:
                st.error("Failed to create text chunks")
                st.stop()
            
            # Get analysis settings
            threshold = similarity_threshold if 'similarity_threshold' in locals() else 0.7
            max_themes = max_themes if 'max_themes' in locals() else 15
            
            # Step 3: Filter relevant chunks
            current_step += 1
            progress_placeholder.progress(current_step / total_steps)
            status_placeholder.info(f"Step {current_step}/{total_steps}: Finding relevant content using AI analysis...")
            
            relevant_chunks = analyzer.filter_relevant_chunks(chunks, all_topics, similarity_threshold=threshold)
            
            # Step 4: Extract themes
            current_step += 1
            progress_placeholder.progress(current_step / total_steps)
            status_placeholder.info(f"Step {current_step}/{total_steps}: Extracting themes and calculating relevance...")
            
            extracted_themes = analyzer.extract_themes_from_chunks(relevant_chunks, all_topics, max_themes=max_themes)
            
            # Step 5: Calculate relationships
            current_step += 1
            progress_placeholder.progress(current_step / total_steps)
            status_placeholder.info(f"Step {current_step}/{total_steps}: Analyzing theme relationships...")
            
            relationship_analysis = calc.calculate_theme_relationships(extracted_themes, chunks)
            
            # Step 6: Prepare visualization
            current_step += 1
            progress_placeholder.progress(current_step / total_steps)
            status_placeholder.info(f"Step {current_step}/{total_steps}: Preparing visualizations...")
            
            viz_data = calc.prepare_visualization_data(extracted_themes, relationship_analysis)
            
            # Complete!
            progress_placeholder.progress(1.0)
            status_placeholder.success("Analysis complete! Explore the results below.")
            
            # Store results in session state for the three sections
            st.session_state.analysis_results = {
                'chunks': chunks,
                'relevant_chunks': relevant_chunks,
                'extracted_themes': extracted_themes,
                'relationship_analysis': relationship_analysis,
                'viz_data': viz_data,
                'research_topics': all_topics,
                'topic_warnings': topic_warnings,
                'document_name': uploaded_file.name
            }
            
        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"Error during analysis: {str(e)}")
    
    # Display results if analysis is complete
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        
        # Analysis Summary Header
        st.markdown("---")
        st.markdown(f"## {get_bootstrap_icon('graph-up')} Analysis Results", unsafe_allow_html=True)
        
        
        # Summary metrics in modern cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                "Document", 
                results['document_name'], 
                "file-earmark-text",
                f"{len(results['chunks'])} total chunks"
            ), unsafe_allow_html=True)
        
        with col2:
            relevance_rate = (len(results['relevant_chunks']) / len(results['chunks'])) * 100 if results['chunks'] else 0
            st.markdown(create_metric_card(
                "Relevant Content", 
                f"{len(results['relevant_chunks'])} chunks", 
                "check-circle",
                f"{relevance_rate:.1f}% relevance rate"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "Themes Extracted", 
                len(results['extracted_themes']), 
                "tags",
                f"{results['viz_data'].get('relationship_count', 0)} relationships"
            ), unsafe_allow_html=True)
        
        with col4:
            avg_confidence = sum(theme.get('confidence', 0) for theme in results['extracted_themes']) / len(results['extracted_themes']) if results['extracted_themes'] else 0
            analysis_method = "AI Analysis" if results['extracted_themes'] and results['extracted_themes'][0].get('source') == 'gpt-4o-mini' else "Keyword Analysis"
            st.markdown(create_metric_card(
                "Analysis Quality", 
                f"{avg_confidence:.2f}", 
                "speedometer2",
                analysis_method
            ), unsafe_allow_html=True)
        
        # Create chunk-theme mapping for tabs
        chunk_theme_mapping = {}
        for chunk in results['chunks']:
            chunk_id = chunk['id']
            chunk_themes = []
            
            # Find themes that appear in this chunk
            for theme in results['extracted_themes']:
                if chunk_id in theme.get('chunk_ids', []):
                    chunk_themes.append({
                        'name': theme['name'],
                        'confidence': theme.get('confidence', 0),
                        'source': theme.get('source', 'unknown'),
                        'description': theme.get('description', 'No description'),
                        'evidence': theme.get('evidence', [])
                    })
            
            if chunk_themes:  # Only store chunks that have themes
                chunk_theme_mapping[chunk_id] = {
                    'text': chunk['text'],
                    'themes': chunk_themes,
                    'relevance_score': chunk.get('relevance_score', 0),
                    'relevance_method': chunk.get('relevance_method', 'unknown'),
                    'position': chunk.get('start_position', chunk_id * 1000)
                }
        
        # Create main tabs
        if results['extracted_themes']:
            # Create tab names - Overview + All Chunks + Visualization + individual chunks with themes
            tab_names = ["Overview", "All Chunks", "Visualization"]
            
            # Add chunk tabs only for chunks that have themes
            sorted_chunk_ids = sorted(chunk_theme_mapping.keys())
            for chunk_id in sorted_chunk_ids:
                theme_count = len(chunk_theme_mapping[chunk_id]['themes'])
                tab_names.append(f"Chunk {chunk_id} ({theme_count} themes)")
            
            # Create the tabs
            tabs = st.tabs(tab_names)
            
            # Overview Tab
            with tabs[0]:
                st.markdown(f"### {get_bootstrap_icon('bullseye')} Extracted Themes Overview", unsafe_allow_html=True)
                st.write("All themes found in the document:")
                
                # Display all extracted themes
                for theme in results['extracted_themes']:
                    with st.expander(f"**{theme['name']}** - Confidence: {theme.get('confidence', 0):.2f}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Description:** {theme.get('description', 'No description')}")
                            st.write(f"**Source:** {theme.get('source', 'unknown')}")
                            st.write(f"**Chunk Frequency:** {theme.get('chunk_frequency', 0)}")
                            
                            # Evidence
                            evidence = theme.get('evidence', [])
                            if evidence:
                                st.write("**Key Evidence:**")
                                for ev in evidence[:3]:
                                    st.write(f"• {ev}")
                        
                        with col2:
                            # Theme metrics
                            theme_metrics = results['relationship_analysis'].get('theme_metrics', {}).get(theme['name'], {})
                            if theme_metrics:
                                st.metric("Centrality", f"{theme_metrics.get('centrality', 0):.3f}")
                                st.metric("Importance", f"{theme_metrics.get('importance', 0):.3f}")
                            
                            # Show which chunks contain this theme
                            chunk_ids = theme.get('chunk_ids', [])
                            if chunk_ids:
                                st.write("**Found in chunks:**")
                                st.write(", ".join([f"Chunk {cid}" for cid in chunk_ids[:5]]))
                
                # Research Topics Analysis
                st.markdown("---")
                st.markdown(f"### {get_bootstrap_icon('search')} Research Topics Analysis", unsafe_allow_html=True)
                st.write("Your input research topics and questions, and how they relate to the extracted themes:")
                
                # Display research topics/questions
                if results['research_topics']:
                    for i, topic in enumerate(results['research_topics']):
                        is_question = "?" in topic
                        icon = "question-circle" if is_question else "tag"
                        topic_type = "Research Question" if is_question else "Research Topic"
                        
                        with st.expander(f"**{topic_type} {i+1}:** {topic}"):
                            # Find themes related to this topic
                            related_themes = []
                            topic_lower = topic.lower()
                            
                            for theme in results['extracted_themes']:
                                theme_name = theme['name']
                                theme_lower = theme_name.lower()
                                
                                # Simple alignment calculation based on word overlap
                                topic_words = set(topic_lower.split())
                                theme_words = set(theme_lower.split())
                                
                                if topic_words and theme_words:
                                    alignment = len(topic_words.intersection(theme_words)) / len(topic_words.union(theme_words))
                                else:
                                    alignment = 0
                                
                                # Also check if topic words appear in theme description
                                description = theme.get('description', '').lower()
                                description_alignment = sum(1 for word in topic_words if word in description) / len(topic_words) if topic_words else 0
                                
                                final_alignment = max(alignment, description_alignment * 0.8)
                                
                                if final_alignment > 0.15:  # Lower threshold for display
                                    related_themes.append({
                                        'name': theme_name,
                                        'confidence': theme.get('confidence', 0),
                                        'alignment': final_alignment,
                                        'description': theme.get('description', 'No description')
                                    })
                            
                            # Sort by alignment score
                            related_themes.sort(key=lambda x: x['alignment'], reverse=True)
                            
                            if related_themes:
                                st.write(f"**Related themes found ({len(related_themes)}):**")
                                for theme in related_themes[:5]:  # Show top 5 related themes
                                    confidence_color = "🟢" if theme['confidence'] > 0.7 else "🟡" if theme['confidence'] > 0.4 else "⚪"
                                    st.write(f"• {confidence_color} **{theme['name']}** (confidence: {theme['confidence']:.2f}, alignment: {theme['alignment']:.2f})")
                                    st.write(f"  *{theme['description'][:100]}{'...' if len(theme['description']) > 100 else ''}*")
                            else:
                                st.write("*No directly related themes found for this topic.*")
                                st.write("This could mean:")
                                st.write("- The document doesn't contain content about this topic")
                                st.write("- The topic needs to be more specific")
                                st.write("- The relevance threshold is too high")
                
                # Analysis summary
                st.write("---")
                st.write(f"**Analysis Summary:**")
                st.write(f"• **{len(results['research_topics'])}** research topics/questions provided")
                st.write(f"• **{len(results['extracted_themes'])}** themes extracted from document")
                
                # Calculate how many topics had related themes
                topics_with_themes = 0
                for topic in results['research_topics']:
                    topic_lower = topic.lower()
                    has_related = False
                    for theme in results['extracted_themes']:
                        theme_lower = theme['name'].lower()
                        topic_words = set(topic_lower.split())
                        theme_words = set(theme_lower.split())
                        if topic_words and theme_words:
                            alignment = len(topic_words.intersection(theme_words)) / len(topic_words.union(theme_words))
                            if alignment > 0.15:
                                has_related = True
                                break
                    if has_related:
                        topics_with_themes += 1
                
                coverage_rate = (topics_with_themes / len(results['research_topics'])) * 100 if results['research_topics'] else 0
                st.write(f"• **{topics_with_themes}/{len(results['research_topics'])}** topics had related themes found ({coverage_rate:.1f}% coverage)")
            
            # All Chunks Tab
            with tabs[1]:
                st.markdown(f"### {get_bootstrap_icon('collection')} All Document Chunks", unsafe_allow_html=True)
                st.write("Complete breakdown of all chunks and their relevance analysis:")
                
                # Create summary table
                chunk_summary_data = []
                for chunk in results['chunks']:
                    chunk_id = chunk['id']
                    is_relevant = chunk_id in [rc['id'] for rc in results['relevant_chunks']]
                    has_themes = chunk_id in chunk_theme_mapping
                    theme_count = len(chunk_theme_mapping[chunk_id]['themes']) if has_themes else 0
                    relevance_score = chunk.get('relevance_score', 0)
                    token_count = chunk.get('token_count', 0)
                    
                    chunk_summary_data.append({
                        'chunk_id': chunk_id,
                        'is_relevant': is_relevant,
                        'has_themes': has_themes,
                        'theme_count': theme_count,
                        'relevance_score': relevance_score,
                        'token_count': token_count,
                        'text_preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                    })
                
                # Display summary
                st.write(f"**Total chunks in document: {len(results['chunks'])}**")
                st.write(f"**Relevant chunks (passed filtering): {len(results['relevant_chunks'])}**")
                st.write(f"**Chunks with themes extracted: {len(chunk_theme_mapping)}**")
                
                # Show each chunk status
                for chunk_data in chunk_summary_data:
                    chunk_id = chunk_data['chunk_id']
                    
                    # Determine status and color
                    if chunk_data['has_themes']:
                        status = f"<span class='status-success'><i class='bi bi-check-circle-fill'></i> Has {chunk_data['theme_count']} themes</span>"
                        status_color = "normal"
                    elif chunk_data['is_relevant']:
                        status = "<span class='status-warning'><i class='bi bi-exclamation-triangle-fill'></i> Relevant but no themes extracted</span>"
                        status_color = "normal"
                    else:
                        status = "<span class='status-error'><i class='bi bi-x-circle-fill'></i> Not relevant (filtered out)</span>"
                        status_color = "normal"
                    
                    # Create clean status text for expander title (no HTML)
                    if chunk_data['has_themes']:
                        status_text = f"Has {chunk_data['theme_count']} themes"
                    elif chunk_data['is_relevant']:
                        status_text = "Relevant but no themes extracted"
                    else:
                        status_text = "Not relevant (filtered out)"
                    
                    with st.expander(f"**Chunk {chunk_id}** - {status_text} (Tokens: {chunk_data['token_count']:,})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**Text Preview:**")
                            st.write(f"*{chunk_data['text_preview']}*")
                        
                        with col2:
                            st.metric("Relevance Score", f"{chunk_data['relevance_score']:.2f}")
                            st.metric("Token Count", f"{chunk_data['token_count']:,}")
                            st.metric("Theme Count", chunk_data['theme_count'])
                            
                            if not chunk_data['is_relevant']:
                                st.write("**Why filtered out:**")
                                if chunk_data['relevance_score'] < 0.7:
                                    st.write(f"Relevance score ({chunk_data['relevance_score']:.2f}) below threshold (0.7)")
                                else:
                                    st.write("No specific reason found")
            
            # Visualization Tab
            with tabs[2]:
                st.markdown(f"### {get_bootstrap_icon('bar-chart')} Theme and Topic Analysis Visualization", unsafe_allow_html=True)
                st.write("Interactive visualizations showing relationships between your research topics and extracted themes:")
                
                try:
                    if results['extracted_themes']:
                        # Prepare theme data
                        theme_data = []
                        for theme in results['extracted_themes']:
                            theme_data.append({
                                'Theme': theme['name'],
                                'Confidence': theme.get('confidence', 0),
                                'Frequency': theme.get('chunk_frequency', 0),
                                'Source': theme.get('source', 'unknown')
                            })
                        
                        # Calculate alignments and coverage
                        alignment_data = calculate_alignment_data(results['research_topics'], results['extracted_themes'])
                        topic_coverage = calculate_topic_coverage(results['research_topics'], results['extracted_themes'])
                        
                        # Chart 1: Interactive Network Graphs
                        st.markdown("#### Theme Relationship Network")
                        
                        # Network visualization choice
                        network_type = st.radio(
                            "Choose network visualization:",
                            ["PyVis (Interactive)", "Plotly (Static)"],
                            horizontal=True
                        )
                        
                        if network_type == "PyVis (Interactive)":
                            # PyVis Network
                            pyvis_html = create_pyvis_network(results['extracted_themes'], results['relationship_analysis'])
                            if pyvis_html and not isinstance(pyvis_html, str) or (isinstance(pyvis_html, str) and not pyvis_html.startswith("Error") and not pyvis_html.startswith("PyVis not")):
                                import streamlit.components.v1 as components
                                components.html(pyvis_html, height=620)
                                st.write("**Interactive Features:** Drag nodes, zoom, hover for details, click to select. Colors indicate confidence levels.")
                            elif isinstance(pyvis_html, str):
                                st.error(pyvis_html)
                                st.info("Falling back to Plotly visualization...")
                                network_fig = create_network_visualization(results['extracted_themes'], results['relationship_analysis'])
                                if network_fig:
                                    st.plotly_chart(network_fig, use_container_width=True)
                            else:
                                st.info("PyVis network not available - insufficient relationship data. Showing Plotly version...")
                                network_fig = create_network_visualization(results['extracted_themes'], results['relationship_analysis'])
                                if network_fig:
                                    st.plotly_chart(network_fig, use_container_width=True)
                        else:
                            # Plotly Network
                            network_fig = create_network_visualization(results['extracted_themes'], results['relationship_analysis'])
                            if network_fig:
                                st.plotly_chart(network_fig, use_container_width=True)
                                st.write("**How to read:** Nodes represent themes (size = frequency + confidence). " +
                                       "Lines show relationships (thickness = strength). Colors indicate confidence levels.")
                            else:
                                st.info("Network visualization not available - insufficient relationship data")
                        
                        # Chart 2: Theme Confidence Distribution
                        st.markdown("#### Theme Confidence Distribution")
                        fig1 = create_theme_confidence_chart(theme_data)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Chart 3: Theme Frequency vs Confidence Scatter
                        st.markdown("#### Theme Frequency vs Confidence Analysis")
                        fig2 = create_frequency_confidence_scatter(theme_data)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Chart 4: Research Topics vs Themes Alignment (if alignments exist)
                        if alignment_data:
                            st.markdown("#### Research Topics vs Extracted Themes Alignment")
                            fig3 = create_alignment_chart(alignment_data)
                            st.plotly_chart(fig3, use_container_width=True)
                            
                            # Chart 5: Topic Coverage Summary
                            st.markdown("#### Research Topic Coverage Summary")
                            fig4 = create_coverage_chart(topic_coverage)
                            st.plotly_chart(fig4, use_container_width=True)
                            
                            # Summary Statistics
                            st.markdown("#### Analysis Summary Statistics")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_topics = len(results['research_topics'])
                                st.metric("Total Research Inputs", total_topics)
                            
                            with col2:
                                covered_topics = len([t for t in topic_coverage if t['Coverage Status'] != 'Poor'])
                                coverage_pct = (covered_topics / total_topics * 100) if total_topics > 0 else 0
                                st.metric("Topics with Themes", f"{covered_topics}/{total_topics}", f"{coverage_pct:.1f}%")
                            
                            with col3:
                                import pandas as pd
                                df_alignment = pd.DataFrame(alignment_data)
                                avg_alignment = df_alignment['Alignment Score'].mean() if not df_alignment.empty else 0
                                st.metric("Avg Topic-Theme Alignment", f"{avg_alignment:.3f}")
                            
                            with col4:
                                import pandas as pd
                                df_themes = pd.DataFrame(theme_data)
                                avg_confidence = df_themes['Confidence'].mean() if not df_themes.empty else 0
                                st.metric("Avg Theme Confidence", f"{avg_confidence:.3f}")
                            
                            # AI-Generated Insights
                            st.markdown("#### Key Insights")
                            insights = generate_insights(topic_coverage, theme_data, alignment_data)
                            
                            for insight in insights:
                                st.write(f"• {insight}")
                        
                        else:
                            st.info("No meaningful alignments found between research topics and extracted themes. This could mean:")
                            st.write("• The document content doesn't match your research focus")
                            st.write("• Research topics need to be more specific")
                            st.write("• Consider adjusting the analysis threshold settings")
                    
                    else:
                        st.warning("No themes available for visualization")
                
                except ImportError:
                    st.error("Visualization libraries not available. Please install plotly, pandas, and networkx.")
                except Exception as e:
                    st.error(f"Error generating visualizations: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging
            
            # Individual Chunk Tabs
            for i, chunk_id in enumerate(sorted_chunk_ids):
                with tabs[i + 3]:  # +3 because tabs are: Overview, All Chunks, Visualization, then individual chunks
                    chunk_data = chunk_theme_mapping[chunk_id]
                    theme_count = len(chunk_data['themes'])
                    relevance = chunk_data['relevance_score']
                    
                    # Chunk header info
                    st.markdown(f"### {get_bootstrap_icon('file-text')} Chunk {chunk_id} Analysis", unsafe_allow_html=True)
                    
                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    with info_col1:
                        st.metric("Themes Found", theme_count)
                    with info_col2:
                        st.metric("Relevance Score", f"{relevance:.2f}")
                    with info_col3:
                        st.metric("Analysis Method", chunk_data['relevance_method'])
                    with info_col4:
                        # Get token count from original chunk data
                        original_chunk = next((c for c in results['chunks'] if c['id'] == chunk_id), None)
                        token_count = original_chunk.get('token_count', 0) if original_chunk else 0
                        st.metric("Token Count", f"{token_count:,}")
                    
                    st.write(f"**Document Position:** ~{chunk_data['position']:,} characters")
                    
                    # Full chunk text
                    st.markdown(f"### {get_bootstrap_icon('card-text')} Full Chunk Text", unsafe_allow_html=True)
                    st.text_area(
                        "Chunk Content", 
                        chunk_data['text'], 
                        height=300, 
                        key=f"chunk_content_{chunk_id}",
                        disabled=True
                    )
                    
                    # Themes found in this chunk
                    st.markdown(f"### {get_bootstrap_icon('tags-fill')} Themes Extracted from this Chunk ({theme_count})", unsafe_allow_html=True)
                    
                    for theme_info in chunk_data['themes']:
                        with st.expander(f"**{theme_info['name']}** - Confidence: {theme_info['confidence']:.2f}"):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.write(f"**Description:** {theme_info['description']}")
                                st.write(f"**Source:** {theme_info['source']}")
                                
                                if theme_info['evidence']:
                                    st.write("**AI-Extracted Evidence:**")
                                    for evidence in theme_info['evidence'][:3]:
                                        st.write(f"• *{evidence}*")
                            
                            with col2:
                                st.write("**Evidence in This Chunk:**")
                                
                                # Find sentences in the chunk that relate to this theme
                                chunk_text = chunk_data['text']
                                theme_related_sentences = []
                                
                                # Split chunk into sentences
                                sentences = chunk_text.replace(".", ".\n").replace("!", "!\n").replace("?", "?\n").split("\n")
                                sentences = [s.strip() for s in sentences if s.strip()]
                                
                                # Look for theme-related content in sentences
                                theme_words = theme_info['name'].lower().split()
                                theme_words = [word for word in theme_words if len(word) > 3]  # Filter short words
                                
                                # Also check evidence words
                                evidence_words = []
                                for evidence in theme_info.get('evidence', []):
                                    evidence_words.extend([word.lower() for word in evidence.split() if len(word) > 3])
                                
                                all_search_words = list(set(theme_words + evidence_words))
                                
                                for sentence in sentences:
                                    sentence_lower = sentence.lower()
                                    # Check if sentence contains theme or evidence words
                                    if any(word in sentence_lower for word in all_search_words):
                                        # Highlight the matching words
                                        highlighted_sentence = sentence
                                        for word in all_search_words:
                                            if word in sentence_lower:
                                                # Simple highlighting with markdown bold
                                                highlighted_sentence = highlighted_sentence.replace(
                                                    word, f"**{word}**"
                                                ).replace(
                                                    word.capitalize(), f"**{word.capitalize()}**"
                                                ).replace(
                                                    word.upper(), f"**{word.upper()}**"
                                                )
                                        theme_related_sentences.append(highlighted_sentence)
                                
                                if theme_related_sentences:
                                    for i, sentence in enumerate(theme_related_sentences[:3]):  # Show top 3
                                        st.markdown(f"<i class='bi bi-quote'></i> {sentence}", unsafe_allow_html=True)
                                        if i < len(theme_related_sentences) - 1:
                                            st.write("")  # Add space between sentences
                                else:
                                    st.write("*No specific evidence sentences found in this chunk*")
                                    
                                    # Fallback: show context around theme words
                                    if theme_words:
                                        st.write("**Theme context:**")
                                        for word in theme_words[:2]:
                                            if word in chunk_text.lower():
                                                # Find context around the word
                                                word_index = chunk_text.lower().find(word)
                                                start = max(0, word_index - 50)
                                                end = min(len(chunk_text), word_index + len(word) + 50)
                                                context = chunk_text[start:end]
                                                if start > 0:
                                                    context = "..." + context
                                                if end < len(chunk_text):
                                                    context = context + "..."
                                                st.write(f"• *{context}*")
                                                break
        
        else:
            st.info("No themes were extracted from the document.")
    
    elif not can_analyze:
        # Show welcome screen and instructions
        st.markdown("## Welcome to Document Theme Analysis!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### How to get started:
            
            1. **Upload a document** (PDF or DOCX) in the sidebar
            2. **Enter research topics** you want to find in the document
            3. **Add research questions** (optional)
            4. **Adjust analysis settings** (optional)
            5. **Click "Start Analysis"** to begin
            
            ### What this tool does:
            - Extracts text from your PDF or DOCX document
            - Identifies themes related to your research topics
            - Shows relationships between different themes
            - Creates interactive visualizations of the results
            """)
        
        with col2:
            st.info("""
            **Tips for better results:**
            
            Enter specific topics and research questions that match what you're looking for in the document.
            
            **Examples of good topics:**
            - Leadership styles
            - Employee satisfaction
            - Digital transformation
            - Customer feedback
            - Innovation processes
            
            **Examples of research questions:**
            - How does remote work affect productivity?
            - What factors drive employee engagement?
            - What are the barriers to innovation?
            """)
        
        # Status indicators
        st.markdown("### Current Status:")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            if uploaded_file is not None:
                st.success("Document uploaded")
            else:
                st.warning("No document uploaded")
        
        with status_col2:
            if len(locals().get('all_topics', [])) > 0:
                st.success(f"{len(locals().get('all_topics', []))} research topics selected")
            else:
                st.warning("No research topics selected")
    
    # Footer
    st.markdown("---")
    st.markdown("**Usage:** Upload a document (PDF or DOCX), enter your research topics, and click analyze to identify and visualize document themes.")

if __name__ == "__main__":
    main()
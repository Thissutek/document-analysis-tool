"""
UI Components Module
Handles Streamlit UI components, Bootstrap styling, and form validation
"""
import streamlit as st
import re
from typing import List, Tuple


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


def create_metric_card(title: str, value: str, icon: str, description: str = None) -> str:
    """
    Create a modern metric card
    
    Args:
        title: Card title
        value: Main value to display
        icon: Bootstrap icon name
        description: Optional description text
        
    Returns:
        str: HTML for the metric card
    """
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


def get_bootstrap_icon(icon_name: str) -> str:
    """
    Return Bootstrap icon HTML
    
    Args:
        icon_name: Name of the Bootstrap icon
        
    Returns:
        str: HTML for the icon
    """
    return f'<i class="bi bi-{icon_name} icon"></i>'


def create_research_analysis_card(research_topics: List[str], topic_warnings: List[str]) -> str:
    """
    Create a card showing the AI's understanding of research topics
    
    Args:
        research_topics: List of research topics
        topic_warnings: List of warning messages
        
    Returns:
        str: HTML for the research analysis card
    """
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


def validate_and_format_topics(topics_list: List[str]) -> Tuple[List[str], List[str]]:
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
        
        # Check for duplicates
        if cleaned_topic not in valid_topics:
            valid_topics.append(cleaned_topic)
        else:
            warnings.append(f"Duplicate topic removed: '{cleaned_topic}'")
    
    return valid_topics, warnings

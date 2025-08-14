"""
Analysis Helpers Module
Handles data analysis calculations for theme-topic alignment, coverage, and insights generation
"""
from typing import List, Dict, Any
import pandas as pd


def calculate_alignment_data(research_topics: List[str], extracted_themes: List[Dict]) -> List[Dict]:
    """
    Calculate topic-theme alignment data
    
    Args:
        research_topics: List of research topics/questions
        extracted_themes: List of extracted theme dictionaries
        
    Returns:
        List[Dict]: Alignment data for visualization
    """
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


def calculate_topic_coverage(research_topics: List[str], extracted_themes: List[Dict]) -> List[Dict]:
    """
    Calculate topic coverage statistics
    
    Args:
        research_topics: List of research topics/questions
        extracted_themes: List of extracted theme dictionaries
        
    Returns:
        List[Dict]: Topic coverage data for visualization
    """
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


def generate_insights(topic_coverage: List[Dict], theme_data: List[Dict], alignment_data: List[Dict]) -> List[str]:
    """
    Generate AI insights based on analysis results
    
    Args:
        topic_coverage: Topic coverage data
        theme_data: Theme data with confidence scores
        alignment_data: Theme-topic alignment data
        
    Returns:
        List[str]: Generated insights for display
    """
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
        df_alignment = pd.DataFrame(alignment_data)
        avg_alignment = df_alignment['Alignment Score'].mean()
        
        if avg_alignment > 0.4:
            insights.append("🔗 Strong alignment between research topics and extracted themes")
        elif avg_alignment > 0.2:
            insights.append("🔍 Moderate alignment - some thematic overlap found")
        else:
            insights.append("❓ Weak alignment - extracted themes may be broader than research focus")
    
    return insights


def calculate_theme_statistics(extracted_themes: List[Dict]) -> Dict[str, Any]:
    """
    Calculate comprehensive theme statistics
    
    Args:
        extracted_themes: List of extracted theme dictionaries
        
    Returns:
        Dict: Theme statistics for analysis
    """
    if not extracted_themes:
        return {
            'total_themes': 0,
            'avg_confidence': 0,
            'avg_frequency': 0,
            'high_confidence_count': 0,
            'source_distribution': {}
        }
    
    # Basic statistics
    total_themes = len(extracted_themes)
    avg_confidence = sum(theme.get('confidence', 0) for theme in extracted_themes) / total_themes
    avg_frequency = sum(theme.get('chunk_frequency', 0) for theme in extracted_themes) / total_themes
    
    # High confidence themes
    high_confidence_count = len([t for t in extracted_themes if t.get('confidence', 0) > 0.7])
    
    # Source distribution
    source_distribution = {}
    for theme in extracted_themes:
        source = theme.get('source', 'unknown')
        source_distribution[source] = source_distribution.get(source, 0) + 1
    
    return {
        'total_themes': total_themes,
        'avg_confidence': avg_confidence,
        'avg_frequency': avg_frequency,
        'high_confidence_count': high_confidence_count,
        'high_confidence_percentage': (high_confidence_count / total_themes) * 100,
        'source_distribution': source_distribution
    }


def analyze_theme_relationships(extracted_themes: List[Dict], relationship_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze theme relationships and calculate network metrics
    
    Args:
        extracted_themes: List of extracted theme dictionaries
        relationship_analysis: Relationship analysis data
        
    Returns:
        Dict: Relationship analysis summary
    """
    if not extracted_themes or not relationship_analysis:
        return {
            'total_relationships': 0,
            'strong_relationships': 0,
            'weak_relationships': 0,
            'avg_relationship_strength': 0,
            'most_connected_theme': None
        }
    
    correlations = relationship_analysis.get('correlations', {})
    theme_metrics = relationship_analysis.get('theme_metrics', {})
    
    # Relationship strength analysis
    total_relationships = len(correlations)
    strong_relationships = len([strength for strength in correlations.values() if strength > 0.6])
    moderate_relationships = len([strength for strength in correlations.values() if 0.3 <= strength <= 0.6])
    weak_relationships = len([strength for strength in correlations.values() if strength < 0.3])
    
    avg_relationship_strength = sum(correlations.values()) / total_relationships if total_relationships > 0 else 0
    
    # Find most connected theme
    most_connected_theme = None
    max_connections = 0
    
    for theme in extracted_themes:
        theme_name = theme['name']
        metrics = theme_metrics.get(theme_name, {})
        connection_count = metrics.get('connection_count', 0)
        
        if connection_count > max_connections:
            max_connections = connection_count
            most_connected_theme = {
                'name': theme_name,
                'connections': connection_count,
                'centrality': metrics.get('centrality', 0),
                'importance': metrics.get('importance', 0)
            }
    
    return {
        'total_relationships': total_relationships,
        'strong_relationships': strong_relationships,
        'moderate_relationships': moderate_relationships,
        'weak_relationships': weak_relationships,
        'avg_relationship_strength': avg_relationship_strength,
        'most_connected_theme': most_connected_theme,
        'relationship_distribution': {
            'strong': strong_relationships,
            'moderate': moderate_relationships,
            'weak': weak_relationships
        }
    }

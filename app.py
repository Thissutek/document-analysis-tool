import streamlit as st
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

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
            placeholder="Example:\nEmployee motivation\nWorkplace productivity\nTeam collaboration\nRemote work challenges",
            height=120,
            help="Enter each topic or theme on a separate line"
        )
        
        # Research questions
        st.write("**Research Questions:**")
        research_questions = st.text_area(
            "What specific questions do you want answered?",
            placeholder="Example:\nHow does leadership style affect team performance?\nWhat factors contribute to employee satisfaction?\nWhat are the main barriers to innovation?",
            height=100,
            help="Enter specific questions you want the analysis to address"
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
        with st.spinner("Analyzing document... This may take a few minutes."):
            try:
                # Display analysis setup
                st.success("Starting document analysis!")
                
                # Show what will be analyzed
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Document", uploaded_file.name)
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                with col2:
                    st.metric("Research Topics", len(all_topics))
                    if 'similarity_threshold' in locals():
                        st.metric("Relevance Threshold", f"{similarity_threshold:.1f}")
                
                with col3:
                    if 'max_themes' in locals():
                        st.metric("Max Themes", max_themes)
                    st.metric("Status", "Processing")
                
                # Display topics being searched
                st.subheader("Topics Being Analyzed:")
                topic_cols = st.columns(3)
                for i, topic in enumerate(all_topics):
                    with topic_cols[i % 3]:
                        st.write(f"• {topic}")
                
                # Step 1: Extract text from document
                st.subheader("Step 1: Document Text Extraction")
                progress_bar = st.progress(0)
                
                parser = DocumentParser()
                extracted_text = parser.extract_text_from_document(uploaded_file)
                progress_bar.progress(20)
                
                if not extracted_text:
                    st.error("Failed to extract text from document")
                    st.stop()
                
                st.success(f"Extracted {len(extracted_text)} characters from document")
                
                # Step 2: Chunk the text
                st.subheader("Step 2: Text Chunking")
                chunker = TextChunker(chunk_tokens=1000, overlap_tokens=100)
                chunks = chunker.chunk_text(extracted_text)
                progress_bar.progress(40)
                
                if not chunks:
                    st.error("Failed to create text chunks")
                    st.stop()
                
                # Display chunking stats
                chunk_stats = chunker.get_chunk_stats(chunks)
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Total Chunks", chunk_stats.get('total_chunks', 0))
                
                with stats_col2:
                    st.metric("Avg Tokens/Chunk", f"{chunk_stats.get('average_tokens_per_chunk', 0):.0f}")
                
                with stats_col3:
                    st.metric("Total Tokens", f"{chunk_stats.get('total_tokens', 0):,}")
                
                with stats_col4:
                    st.metric("Total Words", f"{chunk_stats.get('total_words', 0):,}")
                
                st.success(f"Created {len(chunks)} chunks for analysis")
                
                # Step 3: AI Relevance Filtering
                st.subheader("Step 3: AI Relevance Filtering")
                progress_bar.progress(60)
                
                # Filter chunks based on relevance to research topics using AI
                analyzer = ThemeAnalyzer()
                
                # Use similarity threshold from settings, or default
                threshold = similarity_threshold if 'similarity_threshold' in locals() else 0.7
                
                relevant_chunks = analyzer.filter_relevant_chunks(
                    chunks, 
                    all_topics, 
                    similarity_threshold=threshold
                )
                
                progress_bar.progress(80)
                
                # Display relevance results
                if relevant_chunks:
                    relevance_col1, relevance_col2, relevance_col3 = st.columns(3)
                    
                    with relevance_col1:
                        st.metric("Relevant Chunks", len(relevant_chunks))
                    
                    with relevance_col2:
                        st.metric("Total Chunks", len(chunks))
                    
                    with relevance_col3:
                        relevance_percentage = (len(relevant_chunks) / len(chunks)) * 100
                        st.metric("Relevance Rate", f"{relevance_percentage:.1f}%")
                    
                    st.success(f"AI filtering identified {len(relevant_chunks)} relevant chunks")
                    
                    # Show analysis method used
                    method = relevant_chunks[0].get('relevance_method', 'unknown')
                    if method == 'ai_embedding':
                        st.info("Used AI embeddings for relevance analysis")
                    elif method == 'keyword_matching':
                        st.info("Used keyword matching (AI embeddings unavailable)")
                    
                    # Step 4: Theme Extraction
                    st.subheader("Step 4: Theme Extraction")
                    progress_bar.progress(85)
                    
                    max_themes = max_themes if 'max_themes' in locals() else 15
                    extracted_themes = analyzer.extract_themes_from_chunks(
                        relevant_chunks, 
                        all_topics, 
                        max_themes=max_themes
                    )
                    
                    progress_bar.progress(90)
                    
                    if extracted_themes:
                        st.success(f"Extracted {len(extracted_themes)} themes using {'GPT-4o-mini' if analyzer.has_api_key else 'keyword analysis'}")
                        
                        # Step 5: Calculate Theme Relationships
                        st.subheader("Step 5: Theme Relationship Analysis")
                        
                        calc = RelationshipCalculator()
                        relationship_analysis = calc.calculate_theme_relationships(extracted_themes, chunks)
                        
                        # Prepare visualization data
                        viz_data = calc.prepare_visualization_data(extracted_themes, relationship_analysis)
                        
                        progress_bar.progress(95)
                        
                        # Display theme analysis results
                        theme_col1, theme_col2, theme_col3, theme_col4 = st.columns(4)
                        
                        with theme_col1:
                            st.metric("Extracted Themes", len(extracted_themes))
                        
                        with theme_col2:
                            st.metric("Theme Relationships", viz_data.get('relationship_count', 0))
                        
                        with theme_col3:
                            st.metric("Avg Confidence", f"{viz_data.get('avg_confidence', 0):.2f}")
                        
                        with theme_col4:
                            strong_rels = viz_data.get('strong_relationships', 0)
                            st.metric("Strong Relationships", strong_rels)
                        
                        # Step 6: Create Visualizations
                        st.subheader("Step 6: Theme Relevance Visualization")
                        
                        # Create relevance visualization
                        _display_theme_relevance_charts(extracted_themes, all_topics, viz_data)
                        
                        # Display extracted themes in detail
                        st.subheader("Extracted Themes Details:")
                        for i, theme in enumerate(extracted_themes[:5]):  # Show top 5
                            with st.expander(f"{theme['name']} - Confidence: {theme.get('confidence', 0):.2f}"):
                                st.write(f"**Description:** {theme.get('description', 'No description')}")
                                st.write(f"**Source:** {theme.get('source', 'unknown')}")
                                st.write(f"**Chunk Frequency:** {theme.get('chunk_frequency', 0)}")
                                
                                # Theme metrics from relationship analysis
                                theme_metrics = relationship_analysis.get('theme_metrics', {}).get(theme['name'], {})
                                if theme_metrics:
                                    st.write(f"**Centrality:** {theme_metrics.get('centrality', 0):.3f}")
                                    st.write(f"**Importance:** {theme_metrics.get('importance', 0):.3f}")
                                
                                evidence = theme.get('evidence', [])
                                if evidence:
                                    st.write("**Key Evidence:**")
                                    for j, ev in enumerate(evidence[:3]):
                                        st.write(f"• {ev}")
                        
                        progress_bar.progress(100)
                        st.success("Complete theme analysis finished with interactive visualizations!")
                    
                    else:
                        st.warning("No themes could be extracted from the relevant chunks.")
                        progress_bar.progress(100)
                
                else:
                    st.warning("No relevant chunks found with the current relevance threshold. Try lowering the threshold or using different research topics.")
                    progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
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
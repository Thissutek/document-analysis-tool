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
        st.header("📊 Analysis Results")
        
        # Summary metrics
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Document", results['document_name'])
            st.metric("Total Chunks", len(results['chunks']))
        
        with summary_col2:
            st.metric("Relevant Chunks", len(results['relevant_chunks']))
            relevance_rate = (len(results['relevant_chunks']) / len(results['chunks'])) * 100 if results['chunks'] else 0
            st.metric("Relevance Rate", f"{relevance_rate:.1f}%")
        
        with summary_col3:
            st.metric("Extracted Themes", len(results['extracted_themes']))
            st.metric("Theme Relationships", results['viz_data'].get('relationship_count', 0))
        
        with summary_col4:
            avg_confidence = sum(theme.get('confidence', 0) for theme in results['extracted_themes']) / len(results['extracted_themes']) if results['extracted_themes'] else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            analysis_method = "GPT-4o-mini" if results['extracted_themes'] and results['extracted_themes'][0].get('source') == 'gpt-4o-mini' else "Keyword Analysis"
            st.metric("Analysis Method", analysis_method)
        
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
            # Create tab names - Overview + All Chunks + individual chunks with themes
            tab_names = ["📋 Overview", "📊 All Chunks"]
            
            # Add chunk tabs only for chunks that have themes
            sorted_chunk_ids = sorted(chunk_theme_mapping.keys())
            for chunk_id in sorted_chunk_ids:
                theme_count = len(chunk_theme_mapping[chunk_id]['themes'])
                tab_names.append(f"📄 Chunk {chunk_id} ({theme_count} themes)")
            
            # Create the tabs
            tabs = st.tabs(tab_names)
            
            # Overview Tab
            with tabs[0]:
                st.subheader("🎯 Extracted Themes Overview")
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
                
                # Research Topics vs Themes Summary
                st.subheader("🔍 Research Topics Analysis")
                topic_cols = st.columns(min(3, len(results['research_topics'])))
                for i, topic in enumerate(results['research_topics']):
                    with topic_cols[i % len(topic_cols)]:
                        st.write(f"**{i+1}. {topic}**")
                        
                        # Find themes related to this topic
                        related_themes = []
                        for theme in results['extracted_themes']:
                            if any(word.lower() in theme['name'].lower() for word in topic.split() if len(word) > 3):
                                related_themes.append(theme['name'])
                        
                        if related_themes:
                            st.write("Related themes:")
                            for rt in related_themes[:3]:
                                st.write(f"• {rt}")
                        else:
                            st.write("No directly related themes found")
            
            # All Chunks Tab
            with tabs[1]:
                st.subheader("📊 All Document Chunks")
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
                        status = f"✅ Has {chunk_data['theme_count']} themes"
                        status_color = "normal"
                    elif chunk_data['is_relevant']:
                        status = "🟡 Relevant but no themes extracted"
                        status_color = "normal"
                    else:
                        status = "❌ Not relevant (filtered out)"
                        status_color = "normal"
                    
                    with st.expander(f"**Chunk {chunk_id}** - {status} (Tokens: {chunk_data['token_count']:,})"):
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
            
            # Individual Chunk Tabs
            for i, chunk_id in enumerate(sorted_chunk_ids):
                with tabs[i + 2]:  # +2 because first tab is overview, second is all chunks
                    chunk_data = chunk_theme_mapping[chunk_id]
                    theme_count = len(chunk_data['themes'])
                    relevance = chunk_data['relevance_score']
                    
                    # Chunk header info
                    st.subheader(f"📄 Chunk {chunk_id} Analysis")
                    
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
                    st.subheader("📖 Full Chunk Text")
                    st.text_area(
                        "Chunk Content", 
                        chunk_data['text'], 
                        height=300, 
                        key=f"chunk_content_{chunk_id}",
                        disabled=True
                    )
                    
                    # Themes found in this chunk
                    st.subheader(f"🎯 Themes Extracted from this Chunk ({theme_count})")
                    
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
                                sentences = chunk_text.replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n')
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
                                        st.write(f"📝 {sentence}")
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
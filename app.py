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
                    st.metric("Status", "Ready to Process")
                
                # Display topics being searched
                st.subheader("Topics Being Analyzed:")
                topic_cols = st.columns(3)
                for i, topic in enumerate(all_topics):
                    with topic_cols[i % 3]:
                        st.write(f"• {topic}")
                
                st.info("Full analysis functionality will be implemented in the next steps.")
                
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
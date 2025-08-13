import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Theme Analysis Tool")
    st.markdown("Analyze large text documents to identify themes and visualize their relationships")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("⚠️ OpenAI API key not found. Please set your OPENAI_API_KEY in the .env file.")
        st.info("1. Copy .env.template to .env\n2. Add your OpenAI API key\n3. Restart the application")
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document for theme analysis"
        )
        
        # Research theme input
        research_theme = st.text_area(
            "Research Theme",
            placeholder="Enter keywords, phrases, or questions that define your research focus...",
            help="Describe what themes you're looking for in the document"
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Document", type="primary")
    
    # Main content area
    if uploaded_file is not None and research_theme.strip():
        if analyze_button:
            with st.spinner("Analyzing document... This may take a few minutes."):
                try:
                    # Placeholder for analysis logic
                    st.success("Analysis complete!")
                    st.info("📝 Analysis functionality will be implemented in the source modules.")
                    
                    # Display file info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Document", uploaded_file.name)
                        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                    
                    with col2:
                        st.metric("Research Theme", len(research_theme.split()))
                        st.metric("Status", "Ready for Implementation")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        else:
            st.info("👆 Click 'Analyze Document' to start the analysis")
    
    elif uploaded_file is None:
        st.info("📁 Please upload a PDF document to begin")
    
    elif not research_theme.strip():
        st.info("🎯 Please enter your research theme to focus the analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("**Usage:** Upload a PDF document, enter your research theme, and click analyze to identify and visualize document themes.")

if __name__ == "__main__":
    main()
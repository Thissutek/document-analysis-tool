# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit application for document theme analysis that processes PDF and DOCX documents to identify themes and visualize their relationships through interactive bubble graphs. The project has a complete development environment and input form ready.

## Development Setup

### Environment Setup
1. Virtual environment already created in `venv/`
2. Activate environment: `source venv/bin/activate`
3. Dependencies already installed via: `pip install -r requirements.txt`
4. Create `.env` file: `cp .env.template .env` then add your OpenAI API key
5. Test setup: `python test_setup.py`

### Running the Application
- Start Streamlit app: `streamlit run app.py`
- Default URL: http://localhost:8501
- Alternative port: `streamlit run app.py --server.port 8502`

### Testing
- Run setup validation: `python test_setup.py`
- Test imports: `python -c "import src.document_parser; print('OK')"`

## Planned Architecture

The application follows a modular structure with these key components:

### Core Modules
- `app.py` - Main Streamlit application entry point
- `src/document_parser.py` - PDF and DOCX text extraction using PyPDF2 and python-docx
- `src/text_chunker.py` - Document segmentation with simple for-loop chunking
- `src/theme_analyzer.py` - AI-powered theme extraction using OpenAI GPT-4o-mini
- `src/relationship_calc.py` - Theme correlation analysis and co-occurrence counting
- `src/visualizer.py` - Streamlit visualization with Plotly bubble charts

### Data Flow
1. Document upload (PDF/DOCX) and text extraction
2. Simple fixed-size text chunking with position tracking
3. Combined relevance filtering and theme extraction using OpenAI embeddings + GPT-4o-mini
4. Theme relationship calculation (co-occurrence and correlation)
5. Interactive visualization generation

### Dependencies
```
streamlit>=1.35.0
PyPDF2>=3.0.1
python-docx>=1.1.0
openai>=1.50.0
pandas>=2.2.0
plotly>=5.17.0
numpy>=1.26.0
scikit-learn>=1.5.0
python-dotenv>=1.0.0
```

## Development Notes

- Use simple for loops for text chunking (avoid complex logic)
- Keep data structures Streamlit-friendly
- Focus on combining relevance filtering with theme extraction for efficiency
- All visualization should use Streamlit + Plotly
- API costs: typically $0.10-$2.00 per document analysis

## Implementation Priority

The README outlines a 7-step implementation plan:
1. Document input processing (PDF parsing)
2. Theme input processing (validation/normalization) 
3. Simple text chunking
4. Combined relevance filtering + theme extraction
5. Theme relationship calculation
6. Data preparation for Streamlit
7. Streamlit visualization generation
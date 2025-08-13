# Document Theme Analysis Tool

A Streamlit application that analyzes large text documents (PDFs) to identify themes and visualize their relationships through interactive bubble graphs.

## Overview

This tool helps researchers analyze documents by:
- Extracting text from PDF documents
- Breaking documents into manageable chunks
- Identifying themes related to researcher's input
- Visualizing theme relationships in interactive bubble graphs

## Features

- **Document Processing**: Extract clean text from PDF files
- **Theme Analysis**: AI-powered theme extraction based on researcher input
- **Interactive Visualization**: Bubble graphs showing theme frequency and correlations
- **Streamlit Interface**: User-friendly web interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd document-theme-analysis-tool
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup**
   ```bash
   python scripts/setup.py
   ```

## Required Dependencies

Create a `requirements.txt` file with these packages:

```
streamlit==1.29.0
PyPDF2==3.0.1
openai==1.3.0
pandas==2.1.0
plotly==5.17.0
numpy==1.25.0
scikit-learn==1.3.0
python-dotenv==1.0.0
```

## Environment Setup

1. **Create a `.env` file in the project root:**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Get your OpenAI API key:**
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Copy and paste it into your `.env` file

## Project Structure

```
document-theme-analysis-tool/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore            # Git ignore file
├── README.md             # This file
│
├── src/                   # Core application modules
│   ├── document_parser.py # PDF/DOCX text extraction
│   ├── text_chunker.py    # Token-based document segmentation
│   ├── theme_analyzer.py  # AI theme extraction (GPT-4o-mini)
│   ├── relationship_calc.py # Theme correlation analysis
│   └── visualizer.py      # Interactive visualizations
│
├── tests/                 # Test suite
│   ├── test_document_parser.py
│   ├── test_chunking_pipeline.py
│   ├── test_theme_pipeline.py
│   └── ...
│
├── docs/                  # Documentation
│   ├── README.md         # Detailed documentation
│   ├── CLAUDE.md         # AI development context
│   └── demo/             # Demo materials
│       ├── demo_usage.md
│       └── ...
│
└── scripts/               # Utility scripts
    ├── setup.py          # Environment setup checker
    └── run_tests.py      # Test runner
```

## Running the Application

1. **Make sure your virtual environment is activated**

2. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   - Streamlit will automatically open http://localhost:8501
   - If it doesn't open automatically, navigate to that URL

## Testing

**Run all tests:**
```bash
python scripts/run_tests.py
```

**Run individual test:**
```bash
python tests/test_document_parser.py
```

**Check setup:**
```bash
python scripts/setup.py
```

## Usage

1. **Upload a PDF document** using the file uploader
2. **Enter your research theme** (keywords, phrases, or questions)
3. **Click "Analyze Document"** to start processing
4. **View results** in the interactive bubble graph:
   - Bubble size = theme frequency
   - Bubble connections = theme correlations
   - Hover for details

## Implementation Steps

Follow these steps to build the application:

### Step 1: Document Input Processing
- [ ] Set up PDF parsing with PyPDF2
- [ ] Create text extraction function
- [ ] Add text cleaning and formatting

### Step 2: Theme Input Processing  
- [ ] Create input validation
- [ ] Add text normalization
- [ ] Prepare theme for analysis

### Step 3: Simple Text Chunking
- [ ] Implement fixed-size chunking with for loop
- [ ] Add position tracking
- [ ] Create chunk array structure

### Step 4: Combined Relevance Filtering + Theme Extraction
- [ ] Set up OpenAI embeddings
- [ ] Implement similarity scoring
- [ ] Create GPT-4o-mini integration for theme extraction
- [ ] Combine filtering and extraction logic

### Step 5: Calculate Theme Relationships
- [ ] Implement co-occurrence counting
- [ ] Calculate correlation strengths
- [ ] Measure relationship to research focus

### Step 6: Prepare Data for Streamlit
- [ ] Create DataFrames for themes and relationships
- [ ] Format data for visualization
- [ ] Add metadata for interactive features

### Step 7: Generate Streamlit Visualization
- [ ] Create bubble chart with Plotly
- [ ] Add interactive features
- [ ] Implement Streamlit interface

## Development Notes

- Use simple for loops for chunking (no complex logic)
- Combine steps 4 and 5 from original plan for efficiency
- Focus on Streamlit for all visualization needs
- Keep data structures simple and Streamlit-friendly

## API Costs

- OpenAI embeddings: ~$0.0001 per 1K tokens
- GPT-4o-mini: ~$0.15 per 1M input tokens
- Typical document analysis: $0.10 - $2.00 depending on document size

## Troubleshooting

### Common Issues:

1. **"Module not found" errors**
   - Make sure virtual environment is activated
   - Verify all packages are installed: `pip list`

2. **OpenAI API errors**
   - Check your API key in `.env` file
   - Verify you have credits in your OpenAI account

3. **PDF parsing errors**
   - Ensure PDF is text-based (not scanned images)
   - Try with different PDF files

4. **Streamlit won't start**
   - Check if port 8501 is available
   - Try: `streamlit run app.py --server.port 8502`



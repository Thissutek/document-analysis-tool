# Document Theme Analysis Tool

A powerful AI-driven Streamlit application that analyzes documents to extract themes and provides comprehensive insights with professional visualizations. Built for researchers, analysts, and professionals who need to quickly understand document content and identify key themes.

## 🎯 Overview

This tool leverages advanced AI technology to help users:
- Extract and analyze themes from PDF documents using GPT-4o-mini
- Filter content based on research focus using AI embeddings
- Visualize theme relationships and patterns 
- Calculate accurate API costs for transparency
- Provide actionable insights for research and analysis

## ✨ Key Features

### 📄 **Document Processing**
- **Smart PDF Extraction**: Clean text extraction from PDF files with proper formatting
- **Token-Aware Chunking**: Intelligent document segmentation optimized for AI processing
- **Large Document Support**: Handles documents of varying sizes efficiently

### 🧠 **AI-Powered Analysis**  
- **Enhanced Theme Extraction**: Two-stage prompting system with validation for high-quality results
- **Relevance Filtering**: AI embeddings to focus on content matching your research interests
- **Confidence Scoring**: Themes include confidence metrics and evidence validation
- **Research Alignment**: Measures how well themes align with your specific research goals

### 📊 **Professional Visualizations**
- **Theme Confidence Distribution**: Understand the reliability of extracted themes
- **Frequency Analysis**: See which themes appear most often in your document
- **Research Alignment Charts**: Visualize how themes relate to your research topics
- **Topic Coverage Analysis**: Understand which research areas are well-covered

### 🔧 **User-Friendly Features**
- **Bootstrap Icons**: Clean, professional interface design
- **API Key Management**: Secure sidebar for OpenAI API key configuration with testing
- **Cost Transparency**: Real-time accurate cost estimation for API usage
- **Responsive Design**: Works seamlessly across different screen sizes

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (users provide their own - this is a free tool!)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd document-theme-analysis-tool
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser to http://localhost:8501
   - Or follow the URL provided in the terminal

## 🔑 API Key Setup

This is a **free tool** where users provide their own OpenAI API keys:

### Option 1: Environment Variable (Recommended for Development)
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Option 2: Sidebar Input (Recommended for Users)
1. Open the application
2. Use the sidebar "🔑 API Configuration" section
3. Enter your OpenAI API key
4. Test the connection using the "Test API Key" button

### Getting an OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Copy the key (starts with 'sk-')

## 💡 How to Use

1. **Configure API Access**
   - Enter your OpenAI API key in the sidebar
   - Test the connection to ensure it's working

2. **Upload Document**
   - Use the file uploader to select a PDF document
   - Wait for the text extraction to complete

3. **Define Research Focus**
   - Enter your research topics (comma-separated)
   - Add research questions (optional, one per line)
   - Adjust relevance threshold if needed

4. **Analyze Document**
   - Click "Analyze Document" to start processing
   - Monitor progress through the status indicators
   - Review real-time cost estimates

5. **Explore Results**
   - **Main Dashboard**: Overview of extracted themes with key metrics
   - **Detailed Analysis**: Deep dive into theme evidence and confidence scores
   - **Visualizations**: Interactive charts showing theme patterns and relationships

## 🏗️ Technical Architecture

### Project Structure
```
document-theme-analysis-tool/
├── app.py                      # Main Streamlit application with API key management
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (optional)
├── README.md                  # This documentation
│
├── src/                       # Core application modules
│   ├── theme_analyzer.py      # AI theme extraction with enhanced prompting
│   ├── text_chunker.py        # Token-based document segmentation  
│   ├── relationship_calc.py   # Theme correlation and relationship analysis
│   │
│   ├── ui/                    # User interface components
│   │   └── ui_components.py   # Bootstrap-styled UI elements
│   │
│   ├── visualization/         # Chart and visualization generators
│   │   └── chart_generators.py # Plotly-based professional charts
│   │
│   └── analysis/              # Analysis and insight generation
│       └── analysis_helpers.py # Statistical analysis and insights
```

### Core Technologies
- **Streamlit**: Web application framework
- **OpenAI GPT-4o-mini**: Enhanced theme extraction with validation
- **OpenAI Embeddings**: text-embedding-ada-002 for relevance filtering
- **Plotly**: Interactive professional visualizations
- **Bootstrap Icons**: Clean, professional UI design
- **tiktoken**: Accurate token counting for cost calculation

## 🔬 AI Processing Pipeline

### 1. Document Preparation
- PDF text extraction and cleaning
- Token-based chunking (1000 tokens per chunk, 100 token overlap)
- Chunk metadata and position tracking

### 2. Relevance Filtering
- AI embeddings comparison between chunks and research topics
- Cosine similarity scoring with configurable threshold
- Reduces API costs by processing only relevant content

### 3. Enhanced Theme Extraction
- **Stage 1**: Initial extraction with detailed prompting for specificity
- **Stage 2**: Validation and refinement for quality assurance
- Evidence collection with chunk references
- Confidence scoring and justification

### 4. Analysis & Insights
- Theme frequency and distribution analysis
- Research alignment scoring
- Relationship and correlation analysis
- Statistical insights and recommendations

## 💰 Cost Transparency

### Typical Costs (per document)
- **Small documents** (5-20 pages): $0.10 - $0.50
- **Medium documents** (20-50 pages): $0.50 - $1.50  
- **Large documents** (50+ pages): $1.50 - $5.00

### Cost Factors
- **Document size**: Larger documents require more processing
- **Research focus**: More specific topics = better filtering = lower costs
- **Relevance threshold**: Higher threshold = fewer chunks processed = lower costs

### Real-Time Cost Tracking
The application provides accurate cost estimates based on:
- Actual token counts for embeddings and theme extraction
- Current OpenAI API pricing
- Number of API calls made during processing

## 🔧 Configuration Options

### Relevance Threshold
- **0.7 (Default)**: Balanced accuracy and cost
- **0.8**: Higher precision, lower cost, may miss themes
- **0.6**: Lower precision, higher cost, more comprehensive

### Theme Extraction
- **Maximum themes**: Default 15, configurable based on document complexity
- **Confidence threshold**: Themes below threshold are flagged for review
- **Evidence requirements**: Each theme requires supporting evidence from text

## 🛠️ Development Features

### Enhanced Prompting System
- Specific instructions for theme extraction quality
- Evidence-based validation requirements
- Confidence scoring with justification
- Actionable theme naming conventions

### Error Handling
- Graceful fallback for API failures
- Keyword-based theme extraction backup
- User-friendly error messages
- Connection testing for API keys

### Performance Optimization
- Batch processing for API efficiency
- Smart chunking to minimize API calls
- Relevance filtering to reduce processing
- Caching for repeated analyses

## 🚀 Deployment Ready

This tool is designed for easy deployment and sharing:

### Features for Public Use
- **No server-side API keys**: Users provide their own OpenAI keys
- **Secure key handling**: Keys are session-based, not stored
- **Cost transparency**: Users see exactly what they'll pay
- **Professional UI**: Bootstrap icons and clean design
- **Responsive design**: Works on desktop and mobile

### Deployment Options
- **Streamlit Community Cloud**: Free hosting for public repositories
- **Local deployment**: Run on personal/organizational servers
- **Docker support**: Containerized deployment (add Dockerfile if needed)

## 📈 Use Cases

### Academic Research
- Literature review and theme identification
- Research paper analysis and categorization
- Grant proposal analysis and gap identification

### Business Analysis
- Market research document analysis
- Competitive intelligence theme extraction  
- Customer feedback and survey analysis

### Content Analysis
- Document classification and organization
- Policy document analysis
- Legal document theme extraction

## ⚠️ Important Notes

### API Key Security
- API keys are stored in session state only
- Keys are not logged or permanently stored
- Use environment variables for development
- Always use sidebar input for production/sharing

### Document Requirements
- PDF files must contain extractable text (not scanned images)
- Optimal document size: 5-100 pages
- Large documents (100+ pages) will require higher API costs

### Cost Management
- Start with higher relevance thresholds (0.8) to minimize costs
- Use specific research topics for better filtering
- Monitor the cost estimates before proceeding with analysis

## 🤝 Contributing

This project welcomes contributions! Areas for enhancement:
- Additional document format support (DOCX, TXT)
- Advanced visualization options
- Export functionality for results
- Batch processing capabilities
- Custom theme extraction prompts

## 📄 License

[Add your chosen license here]

## 🆘 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section below
2. Review the cost estimation if experiencing unexpected charges
3. Verify API key configuration and connection

### Common Troubleshooting

**"Invalid API Key" errors**
- Verify your OpenAI API key starts with 'sk-'
- Test the key using the sidebar test button
- Check your OpenAI account has available credits

**High cost estimates**
- Increase the relevance threshold to 0.8
- Use more specific research topics
- Consider processing shorter documents first

**No themes extracted**
- Lower the relevance threshold to 0.6
- Verify your research topics relate to document content
- Check that the PDF contains extractable text

**Slow processing**
- Large documents take more time (proportional to size)
- API rate limits may cause delays
- Consider processing during off-peak hours
#!/usr/bin/env python3
"""
Test script to verify development environment setup
"""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import streamlit
        import openai
        import pandas
        import plotly
        import PyPDF2
        import docx
        import tiktoken
        import numpy
        import sklearn
        import dotenv
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_src_modules():
    """Test that all source modules can be imported"""
    try:
        from src.document_parser import DocumentParser
        from src.text_chunker import TextChunker
        from src.theme_analyzer import ThemeAnalyzer
        from src.relationship_calc import RelationshipCalculator
        from src.visualizer import Visualizer
        print("✅ All source modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Source module import error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if os.path.exists('.env'):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            print("✅ Environment file configured with API key")
            return True
        else:
            print("⚠️  Environment file exists but API key not set")
            return False
    else:
        print("⚠️  No .env file found - copy .env.template to .env and add your API key")
        return False

if __name__ == "__main__":
    print("🧪 Testing development environment setup...\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Source Modules", test_src_modules), 
        ("Environment Config", test_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All tests passed! Development environment is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
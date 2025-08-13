#!/usr/bin/env python3
"""
Setup script for Document Theme Analysis Tool

This script helps set up the development environment and dependencies.
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_virtual_environment():
    """Check if virtual environment exists and is activated"""
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "venv"
    
    if not venv_path.exists():
        print("⚠️  Virtual environment not found at venv/")
        print("   Create one with: python -m venv venv")
        return False
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
        return True
    else:
        print("⚠️  Virtual environment exists but is not activated")
        print("   Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Try importing key dependencies
        import streamlit
        import openai
        import tiktoken
        import pandas
        import numpy
        import sklearn
        print("✅ Key dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install -r requirements.txt")
        return False

def check_environment_file():
    """Check if .env file is configured"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_template = project_root / ".env.template"
    
    if not env_template.exists():
        print("⚠️  .env.template not found")
        return False
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("   Copy .env.template to .env and add your OpenAI API key")
        return False
    
    # Check if API key is configured
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            if "your_openai_api_key_here" in content:
                print("⚠️  .env file exists but API key not configured")
                print("   Add your OpenAI API key to .env file")
                return False
            elif "OPENAI_API_KEY=" in content:
                print("✅ .env file configured")
                return True
            else:
                print("⚠️  .env file exists but OPENAI_API_KEY not found")
                return False
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

def run_basic_test():
    """Run a basic import test"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.document_parser import DocumentParser
        from src.text_chunker import TextChunker
        from src.theme_analyzer import ThemeAnalyzer
        print("✅ Core modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run setup checks"""
    print("🔧 Document Theme Analysis Tool - Setup Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("Environment File", check_environment_file),
        ("Module Imports", run_basic_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    print(f"  ✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Setup complete! You're ready to run the application.")
        print("   Start with: streamlit run app.py")
    else:
        print("⚠️  Some setup issues found. Please address them above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
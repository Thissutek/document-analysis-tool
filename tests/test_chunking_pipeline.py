#!/usr/bin/env python3
"""
Test script for text chunking and AI filtering pipeline
"""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_token_counting():
    """Test tiktoken integration"""
    print("Testing Token Counting...")
    
    try:
        from src.text_chunker import TextChunker
        
        chunker = TextChunker()
        
        test_text = "This is a test sentence. It should be counted properly."
        token_count = chunker.count_tokens(test_text)
        
        print(f"Test text: '{test_text}'")
        print(f"Token count: {token_count}")
        
        if token_count > 0:
            print("✅ Token counting works")
            return True
        else:
            print("❌ Token counting failed")
            return False
            
    except Exception as e:
        print(f"❌ Token counting error: {e}")
        return False

def test_chunking_functionality():
    """Test text chunking with token limits"""
    print("\nTesting Text Chunking...")
    
    try:
        from src.text_chunker import TextChunker
        
        # Create a longer test text
        test_text = """
        This is the first paragraph of our test document. It contains information about leadership styles 
        and their impact on team performance. Leaders who adopt collaborative approaches tend to see 
        better results from their teams.
        
        The second paragraph discusses employee motivation and engagement factors. Research shows that 
        employees are more motivated when they feel valued and have opportunities for growth. This 
        directly relates to workplace productivity and job satisfaction.
        
        Our third paragraph focuses on digital transformation challenges. Many organizations struggle 
        with implementing new technologies while maintaining operational efficiency. Change management 
        becomes crucial in these scenarios.
        
        Finally, this document concludes with thoughts on innovation processes and how they can be 
        improved through better team collaboration and leadership support. Innovation requires both 
        creativity and structured processes to be successful.
        """
        
        # Test chunking with small token limits
        chunker = TextChunker(chunk_tokens=150, overlap_tokens=20)
        chunks = chunker.chunk_text(test_text)
        
        print(f"Original text length: {len(test_text)} characters")
        print(f"Number of chunks created: {len(chunks)}")
        
        if chunks:
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\nChunk {i}:")
                print(f"  Tokens: {chunk.get('token_count', 'N/A')}")
                print(f"  Text preview: {chunk['text'][:100]}...")
            
            # Get statistics
            stats = chunker.get_chunk_stats(chunks)
            print(f"\nChunk Statistics:")
            print(f"  Average tokens per chunk: {stats.get('average_tokens_per_chunk', 0):.1f}")
            print(f"  Total tokens: {stats.get('total_tokens', 0)}")
            
            print("✅ Text chunking works")
            return chunks
        else:
            print("❌ No chunks created")
            return []
            
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        return []

def test_ai_filtering():
    """Test AI relevance filtering (with fallback)"""
    print("\nTesting AI Relevance Filtering...")
    
    try:
        from src.theme_analyzer import ThemeAnalyzer
        
        # Create test chunks
        test_chunks = [
            {
                'id': 0,
                'text': 'This chunk discusses leadership styles and management approaches that affect team dynamics.',
                'token_count': 15
            },
            {
                'id': 1, 
                'text': 'Here we talk about cooking recipes and food preparation techniques for dinner parties.',
                'token_count': 14
            },
            {
                'id': 2,
                'text': 'Employee motivation and workplace productivity are key factors in organizational success.',
                'token_count': 13
            },
            {
                'id': 3,
                'text': 'The weather forecast shows rain and storms coming this weekend for outdoor activities.',
                'token_count': 14
            }
        ]
        
        # Test topics that should match some chunks
        research_topics = [
            'leadership effectiveness',
            'employee motivation', 
            'team performance'
        ]
        
        analyzer = ThemeAnalyzer()
        
        print(f"Test chunks: {len(test_chunks)}")
        print(f"Research topics: {research_topics}")
        
        # Note: This may use fallback keyword matching if OpenAI API is not configured
        relevant_chunks = analyzer.filter_relevant_chunks(
            test_chunks, 
            research_topics, 
            similarity_threshold=0.3  # Lower threshold for testing
        )
        
        print(f"Relevant chunks found: {len(relevant_chunks)}")
        
        for chunk in relevant_chunks:
            method = chunk.get('relevance_method', 'unknown')
            score = chunk.get('relevance_score', 0)
            print(f"  Chunk {chunk['id']}: {score:.2f} ({method})")
            print(f"    Text: {chunk['text'][:80]}...")
        
        if len(relevant_chunks) > 0:
            print("✅ AI filtering works (found relevant chunks)")
            return True
        else:
            print("⚠️  No relevant chunks found (may need to adjust threshold or topics)")
            return False
            
    except Exception as e:
        print(f"❌ AI filtering error: {e}")
        return False

def test_integration():
    """Test the complete pipeline integration"""
    print("\nTesting Complete Pipeline Integration...")
    
    try:
        from src.text_chunker import TextChunker
        from src.theme_analyzer import ThemeAnalyzer
        
        # Sample document text
        document_text = """
        Leadership Development and Team Performance Analysis
        
        Effective leadership is crucial for organizational success. This study examines various leadership 
        styles and their impact on team performance metrics. We analyzed data from 500 organizations 
        across different industries to understand the correlation between leadership approaches and 
        employee engagement levels.
        
        Our research indicates that transformational leadership styles yield the highest levels of 
        employee motivation and job satisfaction. Teams led by transformational leaders show 25% 
        higher productivity rates compared to those under traditional management approaches.
        
        Employee motivation factors include recognition, career development opportunities, and 
        workplace autonomy. Organizations that prioritize these factors see significant improvements 
        in retention rates and overall performance metrics.
        
        Digital transformation initiatives require strong change management capabilities. Leaders 
        who can effectively communicate vision and provide support during technological transitions 
        are more likely to achieve successful digital transformation outcomes.
        """
        
        research_topics = [
            'leadership effectiveness',
            'employee motivation',
            'team performance',
            'digital transformation'
        ]
        
        print("Step 1: Chunking document...")
        chunker = TextChunker(chunk_tokens=200, overlap_tokens=30)
        chunks = chunker.chunk_text(document_text)
        
        print(f"  Created {len(chunks)} chunks")
        
        print("Step 2: AI relevance filtering...")
        analyzer = ThemeAnalyzer()
        relevant_chunks = analyzer.filter_relevant_chunks(
            chunks, research_topics, similarity_threshold=0.4
        )
        
        print(f"  Found {len(relevant_chunks)} relevant chunks")
        
        if chunks and relevant_chunks:
            relevance_rate = (len(relevant_chunks) / len(chunks)) * 100
            print(f"  Relevance rate: {relevance_rate:.1f}%")
            print("✅ Complete pipeline integration works")
            return True
        else:
            print("❌ Pipeline integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Text Chunking and AI Filtering Pipeline")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Token Counting", test_token_counting),
        ("Text Chunking", test_chunking_functionality),
        ("AI Filtering", test_ai_filtering),
        ("Pipeline Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Chunking and filtering pipeline is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
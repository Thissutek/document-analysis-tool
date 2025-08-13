#!/usr/bin/env python3
"""
Test script for complete theme analysis pipeline with GPT-4o-mini
"""

def test_theme_extraction():
    """Test theme extraction from relevant chunks"""
    print("Testing Theme Extraction...")
    
    try:
        from src.theme_analyzer import ThemeAnalyzer
        
        # Create test relevant chunks
        test_chunks = [
            {
                'id': 0,
                'text': 'Leadership effectiveness in organizations depends on communication skills and the ability to motivate teams. Strong leaders create clear vision and provide guidance during challenging times.',
                'token_count': 25,
                'relevance_score': 0.8
            },
            {
                'id': 1,
                'text': 'Employee motivation is driven by recognition, career development opportunities, and workplace autonomy. Companies that invest in their people see higher retention rates.',
                'token_count': 24,
                'relevance_score': 0.9
            },
            {
                'id': 2,
                'text': 'Digital transformation requires strong change management and leadership support. Organizations must balance innovation with operational stability during technology transitions.',
                'token_count': 22,
                'relevance_score': 0.7
            }
        ]
        
        research_topics = ['leadership effectiveness', 'employee motivation', 'digital transformation']
        
        analyzer = ThemeAnalyzer()
        print(f"Analyzer has API key: {analyzer.has_api_key}")
        
        # Test theme extraction (will use fallback if no API key)
        themes = analyzer.extract_themes_from_chunks(
            test_chunks, 
            research_topics, 
            max_themes=10
        )
        
        print(f"Extracted {len(themes)} themes")
        
        for theme in themes:
            print(f"  Theme: {theme['name']}")
            print(f"    Confidence: {theme.get('confidence', 0):.2f}")
            print(f"    Source: {theme.get('source', 'unknown')}")
            print(f"    Chunk frequency: {theme.get('chunk_frequency', 0)}")
        
        if themes:
            print("✅ Theme extraction works")
            return themes
        else:
            print("⚠️  No themes extracted")
            return []
            
    except Exception as e:
        print(f"❌ Theme extraction error: {e}")
        return []

def test_relationship_calculation():
    """Test theme relationship calculation"""
    print("\nTesting Theme Relationship Calculation...")
    
    try:
        from src.relationship_calc import RelationshipCalculator
        
        # Create test themes with overlapping chunk_ids
        test_themes = [
            {
                'name': 'Leadership Skills',
                'description': 'Themes related to leadership effectiveness',
                'chunk_ids': [0, 2],  # Appears in chunks 0 and 2
                'chunk_frequency': 2,
                'confidence': 0.8
            },
            {
                'name': 'Employee Engagement',
                'description': 'Themes about employee motivation',
                'chunk_ids': [1],  # Appears in chunk 1
                'chunk_frequency': 1,
                'confidence': 0.9
            },
            {
                'name': 'Change Management',
                'description': 'Themes about organizational change',
                'chunk_ids': [2],  # Appears in chunk 2 (overlaps with Leadership)
                'chunk_frequency': 1,
                'confidence': 0.7
            }
        ]
        
        test_chunks = [{'id': i} for i in range(3)]  # Simple chunk references
        
        calc = RelationshipCalculator()
        
        # Calculate comprehensive relationships
        relationship_analysis = calc.calculate_theme_relationships(test_themes, test_chunks)
        
        print("Relationship Analysis Results:")
        print(f"  Total themes: {relationship_analysis['total_themes']}")
        
        # Show co-occurrence
        cooccurrence = relationship_analysis['cooccurrence']
        print(f"  Co-occurrence pairs: {len(cooccurrence)}")
        for (theme1, theme2), count in cooccurrence.items():
            print(f"    {theme1} <-> {theme2}: {count} shared chunks")
        
        # Show correlations
        correlations = relationship_analysis['correlations']
        print(f"  Correlation pairs: {len(correlations)}")
        for (theme1, theme2), strength in correlations.items():
            print(f"    {theme1} <-> {theme2}: {strength:.3f} strength")
        
        # Show theme metrics
        theme_metrics = relationship_analysis['theme_metrics']
        print(f"  Theme metrics calculated for {len(theme_metrics)} themes")
        for theme_name, metrics in theme_metrics.items():
            print(f"    {theme_name}:")
            print(f"      Centrality: {metrics['centrality']:.3f}")
            print(f"      Importance: {metrics['importance']:.3f}")
        
        print("✅ Relationship calculation works")
        return relationship_analysis
        
    except Exception as e:
        print(f"❌ Relationship calculation error: {e}")
        return {}

def test_visualization_data_prep():
    """Test visualization data preparation"""
    print("\nTesting Visualization Data Preparation...")
    
    try:
        from src.relationship_calc import RelationshipCalculator
        
        # Use results from previous test
        test_themes = [
            {
                'name': 'Leadership Skills',
                'description': 'Themes related to leadership effectiveness',
                'chunk_ids': [0, 2],
                'chunk_frequency': 2,
                'confidence': 0.8,
                'evidence': ['strong communication', 'team motivation']
            },
            {
                'name': 'Employee Engagement', 
                'description': 'Themes about employee motivation',
                'chunk_ids': [1],
                'chunk_frequency': 1,
                'confidence': 0.9,
                'evidence': ['recognition programs', 'career development']
            }
        ]
        
        calc = RelationshipCalculator()
        relationship_analysis = calc.calculate_theme_relationships(test_themes, [])
        
        # Prepare visualization data
        viz_data = calc.prepare_visualization_data(test_themes, relationship_analysis)
        
        print("Visualization Data Structure:")
        print(f"  Nodes (themes): {len(viz_data['nodes'])}")
        print(f"  Edges (relationships): {len(viz_data['edges'])}")
        print(f"  Theme count: {viz_data['theme_count']}")
        print(f"  Relationship count: {viz_data['relationship_count']}")
        
        # Show node structure
        if viz_data['nodes']:
            sample_node = viz_data['nodes'][0]
            print(f"  Sample node structure: {list(sample_node.keys())}")
        
        # Show edge structure
        if viz_data['edges']:
            sample_edge = viz_data['edges'][0] 
            print(f"  Sample edge structure: {list(sample_edge.keys())}")
        
        # Show summary stats
        summary = viz_data.get('summary_stats', {})
        print(f"  Summary stats: {summary}")
        
        print("✅ Visualization data preparation works")
        return viz_data
        
    except Exception as e:
        print(f"❌ Visualization prep error: {e}")
        return {}

def test_complete_pipeline():
    """Test the complete theme analysis pipeline"""
    print("\nTesting Complete Theme Analysis Pipeline...")
    
    try:
        from src.text_chunker import TextChunker
        from src.theme_analyzer import ThemeAnalyzer
        from src.relationship_calc import RelationshipCalculator
        
        # Sample document
        document_text = """
        Leadership effectiveness is crucial for organizational success. Research shows that transformational 
        leaders who focus on employee motivation and development achieve better team performance outcomes. 
        
        Employee motivation factors include recognition, career growth opportunities, and meaningful work. 
        Organizations that prioritize employee engagement see improved productivity and lower turnover rates.
        
        Digital transformation initiatives require strong leadership and change management capabilities. 
        Successful digital leaders balance innovation with operational stability while maintaining employee morale.
        
        Team performance metrics improve when leadership styles align with organizational culture and values. 
        The most effective leaders adapt their approach based on situational needs and team dynamics.
        """
        
        research_topics = ['leadership effectiveness', 'employee motivation', 'team performance']
        
        print("Pipeline Step 1: Text Chunking")
        chunker = TextChunker(chunk_tokens=100, overlap_tokens=20)
        chunks = chunker.chunk_text(document_text)
        print(f"  Created {len(chunks)} chunks")
        
        print("Pipeline Step 2: Relevance Filtering")
        analyzer = ThemeAnalyzer()
        relevant_chunks = analyzer.filter_relevant_chunks(
            chunks, research_topics, similarity_threshold=0.3
        )
        print(f"  Found {len(relevant_chunks)} relevant chunks")
        
        print("Pipeline Step 3: Theme Extraction")
        themes = analyzer.extract_themes_from_chunks(
            relevant_chunks, research_topics, max_themes=8
        )
        print(f"  Extracted {len(themes)} themes")
        
        print("Pipeline Step 4: Relationship Analysis")
        calc = RelationshipCalculator()
        relationship_analysis = calc.calculate_theme_relationships(themes, chunks)
        print(f"  Calculated relationships for {relationship_analysis['total_themes']} themes")
        
        print("Pipeline Step 5: Visualization Prep")
        viz_data = calc.prepare_visualization_data(themes, relationship_analysis)
        print(f"  Prepared {viz_data['theme_count']} nodes and {viz_data['relationship_count']} edges")
        
        # Summary
        if themes and viz_data:
            print("\n📊 Pipeline Summary:")
            print(f"  Document length: {len(document_text)} characters")
            print(f"  Text chunks: {len(chunks)}")
            print(f"  Relevant chunks: {len(relevant_chunks)}")
            print(f"  Extracted themes: {len(themes)}")
            print(f"  Theme relationships: {viz_data['relationship_count']}")
            print(f"  Avg theme confidence: {viz_data.get('avg_confidence', 0):.2f}")
            
            print("✅ Complete pipeline integration works")
            return True
        else:
            print("❌ Pipeline produced no results")
            return False
            
    except Exception as e:
        print(f"❌ Complete pipeline error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Complete Theme Analysis Pipeline")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Theme Extraction", test_theme_extraction),
        ("Relationship Calculation", test_relationship_calculation), 
        ("Visualization Data Prep", test_visualization_data_prep),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            success = bool(result) or result is True
            results.append(success)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("THEME ANALYSIS TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Theme analysis pipeline with GPT-4o-mini is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    print(f"\nNote: Using GPT-4o-mini model for cost-effective theme extraction.")
    print("Fallback keyword analysis available when API is not accessible.")
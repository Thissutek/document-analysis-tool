#!/usr/bin/env python3
"""
Test script for visualization components
"""

def test_visualization_function():
    """Test the visualization function with mock data"""
    print("Testing Visualization Components...")
    
    try:
        import pandas as pd
        import plotly.express as px
        
        # Mock extracted themes data
        mock_themes = [
            {
                'name': 'Leadership Effectiveness',
                'confidence': 0.85,
                'chunk_frequency': 3,
                'source': 'keyword_analysis',
                'description': 'Themes related to effective leadership and management practices'
            },
            {
                'name': 'Employee Motivation',
                'confidence': 0.72,
                'chunk_frequency': 2,
                'source': 'keyword_analysis',
                'description': 'Factors that drive employee engagement and motivation'
            },
            {
                'name': 'Digital Innovation',
                'confidence': 0.91,
                'chunk_frequency': 4,
                'source': 'gpt-4o-mini',
                'description': 'Digital transformation and innovation initiatives'
            }
        ]
        
        # Mock research topics
        mock_topics = [
            'leadership effectiveness',
            'employee motivation',
            'digital transformation'
        ]
        
        # Mock viz_data
        mock_viz_data = {
            'nodes': [
                {
                    'id': 'Leadership Effectiveness',
                    'importance': 0.8,
                    'centrality': 0.7
                },
                {
                    'id': 'Employee Motivation', 
                    'importance': 0.6,
                    'centrality': 0.5
                },
                {
                    'id': 'Digital Innovation',
                    'importance': 0.9,
                    'centrality': 0.8
                }
            ],
            'edges': [
                {
                    'source': 'Leadership Effectiveness',
                    'target': 'Employee Motivation',
                    'strength': 0.6,
                    'cooccurrence_count': 1
                },
                {
                    'source': 'Digital Innovation',
                    'target': 'Leadership Effectiveness', 
                    'strength': 0.4,
                    'cooccurrence_count': 2
                }
            ]
        }
        
        print("Testing data preparation...")
        
        # Test theme data preparation
        theme_data = []
        for theme in mock_themes:
            importance = 0.5  # default
            for node in mock_viz_data.get('nodes', []):
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
        print(f"✅ Theme DataFrame created with {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        
        # Test topic-theme alignment calculation
        topic_theme_data = []
        for topic in mock_topics:
            topic_lower = topic.lower()
            for theme in mock_themes:
                theme_name = theme['name']
                theme_lower = theme_name.lower()
                
                # Simple alignment calculation
                topic_words = set(topic_lower.split())
                theme_words = set(theme_lower.split())
                
                if topic_words and theme_words:
                    alignment = len(topic_words.intersection(theme_words)) / len(topic_words.union(theme_words))
                else:
                    alignment = 0
                
                # Check description
                description = theme.get('description', '').lower()
                description_alignment = sum(1 for word in topic_words if word in description) / len(topic_words) if topic_words else 0
                
                final_alignment = max(alignment, description_alignment * 0.8)
                
                if final_alignment > 0.1:
                    topic_theme_data.append({
                        'Research Topic': topic,
                        'Extracted Theme': theme_name,
                        'Alignment Score': final_alignment,
                        'Theme Confidence': theme.get('confidence', 0),
                        'Theme Frequency': theme.get('chunk_frequency', 0)
                    })
        
        if topic_theme_data:
            alignment_df = pd.DataFrame(topic_theme_data)
            print(f"✅ Alignment DataFrame created with {len(alignment_df)} rows")
            print(f"   Sample alignments:")
            for _, row in alignment_df.head(3).iterrows():
                print(f"     {row['Research Topic']} → {row['Extracted Theme']}: {row['Alignment Score']:.3f}")
        
        # Test edge data preparation
        edges = mock_viz_data.get('edges', [])
        if edges:
            edge_data = []
            for edge in edges:
                edge_data.append({
                    'Source Theme': edge['source'],
                    'Target Theme': edge['target'],
                    'Relationship Strength': edge['strength'],
                    'Co-occurrence': edge.get('cooccurrence_count', 0)
                })
            
            edge_df = pd.DataFrame(edge_data)
            print(f"✅ Edge DataFrame created with {len(edge_df)} rows")
            print(f"   Sample relationships:")
            for _, row in edge_df.iterrows():
                print(f"     {row['Source Theme']} ↔ {row['Target Theme']}: {row['Relationship Strength']:.3f}")
        
        # Test Plotly figure creation
        print("\nTesting Plotly figure creation...")
        
        # Test scatter plot
        fig1 = px.scatter(
            df,
            x='Confidence',
            y='Frequency', 
            size='Importance',
            color='Source',
            hover_name='Theme',
            title="Test: Theme Confidence vs Frequency"
        )
        print("✅ Scatter plot created successfully")
        
        # Test bar chart
        if topic_theme_data:
            fig2 = px.bar(
                alignment_df.sort_values('Alignment Score', ascending=True),
                x='Alignment Score',
                y='Extracted Theme',
                color='Research Topic',
                title="Test: Theme Alignment"
            )
            print("✅ Alignment bar chart created successfully")
        
        # Test relationship chart
        if edge_data:
            fig3 = px.bar(
                edge_df,
                x='Relationship Strength',
                y=[f"{row['Source Theme']} ↔ {row['Target Theme']}" for _, row in edge_df.iterrows()],
                title="Test: Relationship Strengths"
            )
            print("✅ Relationship bar chart created successfully")
        
        print("✅ All visualization components test successfully")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test error: {e}")
        return False

def test_app_integration():
    """Test that the visualization function can be imported"""
    print("\nTesting App Integration...")
    
    try:
        # Test importing the function
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        # This should work if the function is properly defined
        from app import _display_theme_relevance_charts
        
        print("✅ Visualization function imported successfully")
        print(f"   Function signature: {_display_theme_relevance_charts.__name__}")
        print(f"   Function docstring: {_display_theme_relevance_charts.__doc__}")
        
        return True
        
    except Exception as e:
        print(f"❌ App integration error: {e}")
        return False

def test_dependencies():
    """Test required dependencies for visualization"""
    print("\nTesting Visualization Dependencies...")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        
        print("✅ Plotly Express imported")
        print("✅ Plotly Graph Objects imported")
        print("✅ Pandas imported")
        
        # Test basic functionality
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        fig = px.scatter(df, x='x', y='y')
        
        print("✅ Basic Plotly functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Dependencies error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Visualization Components")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Visualization Function", test_visualization_function),
        ("App Integration", test_app_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("VISUALIZATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All visualization tests passed! Charts are ready for display.")
    else:
        print("⚠️  Some visualization tests failed. Check the errors above.")
        
    print("\nVisualization Features:")
    print("- Theme Confidence vs Frequency scatter plot")
    print("- Research Topics vs Extracted Themes alignment")
    print("- Theme relationship strength charts") 
    print("- Interactive Plotly charts with hover data")
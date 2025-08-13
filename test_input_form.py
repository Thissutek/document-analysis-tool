#!/usr/bin/env python3
"""
Test script for the research topics input form functionality
"""
from app import validate_and_format_topics

def test_topic_validation():
    """Test various topic validation scenarios"""
    print("🧪 Testing Topic Validation Function\n")
    
    test_cases = [
        {
            "name": "Normal topics",
            "input": ["leadership", "innovation", "employee engagement"],
            "expected_count": 3
        },
        {
            "name": "Mixed case and formatting",
            "input": ["  LEADERSHIP  ", "innovation", "Employee Engagement"],
            "expected_count": 3
        },
        {
            "name": "Duplicates and short topics",
            "input": ["leadership", "Leadership", "LEADERSHIP", "ab", "innovation"],
            "expected_count": 2  # Only leadership and innovation should remain
        },
        {
            "name": "Research questions",
            "input": [
                "How does leadership affect team performance?",
                "What are the main challenges in digital transformation?",
                "Why do employees leave companies?"
            ],
            "expected_count": 3
        },
        {
            "name": "Empty and invalid inputs",
            "input": ["", "   ", "ab", "x"],
            "expected_count": 0  # All should be filtered out
        },
        {
            "name": "Long topics",
            "input": ["This is a very long topic that exceeds the maximum length limit and should be truncated to fit within the 200 character limit that we have set for individual research topics and themes" * 2],
            "expected_count": 1  # Should be truncated but kept
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        valid_topics, warnings = validate_and_format_topics(test_case['input'])
        
        print(f"Valid topics ({len(valid_topics)}):")
        for j, topic in enumerate(valid_topics, 1):
            print(f"  {j}. {topic}")
        
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        # Check if result matches expectation
        if len(valid_topics) == test_case['expected_count']:
            print("✅ PASS")
        else:
            print(f"❌ FAIL - Expected {test_case['expected_count']} topics, got {len(valid_topics)}")
        
        print("-" * 50)

def test_common_research_topics():
    """Test with common research topics"""
    print("\n🎯 Testing Common Research Topics\n")
    
    common_topics = [
        "Leadership styles and effectiveness",
        "Employee motivation and engagement", 
        "Digital transformation challenges",
        "Customer satisfaction metrics",
        "Innovation in product development",
        "Remote work productivity",
        "Team collaboration tools",
        "Performance management systems",
        "Organizational culture change",
        "Data-driven decision making"
    ]
    
    valid_topics, warnings = validate_and_format_topics(common_topics)
    
    print(f"Processed {len(common_topics)} common research topics:")
    for i, topic in enumerate(valid_topics, 1):
        print(f"{i:2d}. {topic}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"⚠️  {warning}")
    
    print(f"\n✅ Successfully processed {len(valid_topics)}/{len(common_topics)} topics")

if __name__ == "__main__":
    print("🧪 Testing Research Topics Input Form\n")
    print("="*60)
    
    test_topic_validation()
    test_common_research_topics()
    
    print("\n🎉 All input form tests completed!")
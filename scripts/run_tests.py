#!/usr/bin/env python3
"""
Test runner script for Document Theme Analysis Tool

This script runs all tests in the tests/ directory and provides
a summary of results.
"""
import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run all tests and display results"""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return 1
    
    # Change to project root for imports to work
    os.chdir(project_root)
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    print("🧪 Running Document Theme Analysis Tool Tests")
    print("=" * 50)
    
    # Find all test files
    test_files = list(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("⚠️  No test files found in tests/ directory")
        return 0
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  • {test_file.name}")
    
    print("\n" + "=" * 50)
    
    # Run each test file
    passed = 0
    failed = 0
    
    for test_file in test_files:
        print(f"\n🔍 Running {test_file.name}...")
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {test_file.name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_file.name} - FAILED")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file.name} - TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"💥 {test_file.name} - ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed / (passed + failed) * 100:.1f}%" if (passed + failed) > 0 else "  📈 Success Rate: N/A")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
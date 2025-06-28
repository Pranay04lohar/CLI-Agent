#!/usr/bin/env python3
"""
Simple evaluation script to test the CLI agent performance
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_agent(instruction):
    """Test the agent with a given instruction"""
    try:
        result = subprocess.run(
            [sys.executable, "../src/agent.py", instruction],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "instruction": instruction,
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {
            "instruction": instruction,
            "success": False,
            "output": "",
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "instruction": instruction,
            "success": False,
            "output": "",
            "error": str(e)
        }

def main():
    # Test cases
    test_cases = [
        "Create a new Git branch and switch to it",
        "List all files in current directory",
        "Initialize a new Git repository",
        "Install a Python package using pip",
        "Create a new directory",
        "Copy a file to another location",
        "Check Git status",
        "Compress a folder using tar",
        "Search for text in files using grep",
        "Run a Python script"
    ]
    
    print("🚀 Testing CLI Agent Performance\n")
    print("=" * 50)
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case}")
        result = test_agent(test_case)
        results.append(result)
        
        if result["success"]:
            print(f"✅ Output: {result['output']}")
        else:
            print(f"❌ Error: {result['error']}")
            if result["output"]:
                print(f"   Output: {result['output']}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 SUMMARY: {successful}/{total} tests passed ({successful/total*100:.1f}%)")
    
    # Save detailed results
    os.makedirs("../logs", exist_ok=True)
    with open("../logs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("📁 Detailed results saved to logs/evaluation_results.json")

if __name__ == "__main__":
    main() 
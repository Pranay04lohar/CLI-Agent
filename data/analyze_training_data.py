#!/usr/bin/env python3
"""
Analyze the training data to understand why the fine-tuned model isn't working well
"""

import json
import re
from collections import Counter

def analyze_training_data():
    # Load the cleaned training data
    with open("data/command_qa_cleaned.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("🔍 Training Data Analysis")
    print("=" * 50)
    
    # Basic statistics
    print(f"📊 Total Q&A pairs: {len(data)}")
    
    # Analyze question types
    command_keywords = ['git', 'bash', 'shell', 'command', 'terminal', 'cli', 'run', 'execute']
    direct_command_questions = 0
    explanation_questions = 0
    
    # Analyze answer formats
    step_by_step_answers = 0
    code_block_answers = 0
    direct_command_answers = 0
    verbose_explanation_answers = 0
    
    # Common commands in answers
    command_patterns = []
    
    for item in data:
        question = item["question"].lower()
        answer = item["answer"]
        
        # Question analysis
        if any(keyword in question for keyword in command_keywords):
            if any(phrase in question for phrase in ["how to", "how do", "how can"]):
                direct_command_questions += 1
            else:
                explanation_questions += 1
        
        # Answer analysis
        answer_lines = answer.split('\n')
        answer_length = len(answer)
        
        if answer_length > 1000:
            verbose_explanation_answers += 1
        
        if '```' in answer or answer.count('`') > 4:
            code_block_answers += 1
        
        if any(line.strip().startswith(str(i)) for line in answer_lines for i in range(1, 10)):
            step_by_step_answers += 1
        
        # Extract commands
        git_commands = re.findall(r'git [a-zA-Z-]+', answer)
        command_patterns.extend(git_commands)
    
    print(f"\n📋 Question Types:")
    print(f"   Direct command questions: {direct_command_questions}")
    print(f"   Explanation questions: {explanation_questions}")
    
    print(f"\n📝 Answer Formats:")
    print(f"   Step-by-step answers: {step_by_step_answers}")
    print(f"   Code block answers: {code_block_answers}")
    print(f"   Direct command answers: {direct_command_answers}")
    print(f"   Verbose explanations (>1000 chars): {verbose_explanation_answers}")
    
    # Most common commands
    common_commands = Counter(command_patterns).most_common(10)
    print(f"\n⚡ Most Common Git Commands:")
    for cmd, count in common_commands:
        print(f"   {cmd}: {count} times")
    
    # Sample problematic examples
    print(f"\n⚠️  Potential Issues:")
    print(f"   • {verbose_explanation_answers}/{len(data)} answers are very verbose")
    print(f"   • Training data focuses on explanations, not direct commands")
    print(f"   • Q&A format doesn't match simple command generation task")
    
    # Show a sample Q&A to illustrate the mismatch
    print(f"\n📖 Sample Training Data (showing format mismatch):")
    sample = data[1]  # Take the second item
    print(f"Question: {sample['question'][:100]}...")
    print(f"Answer: {sample['answer'][:200]}...")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Create training data specifically for command generation")
    print(f"   2. Use format: 'Instruction' -> 'Direct Commands'")
    print(f"   3. Focus on concise, actionable responses")
    print(f"   4. Include examples like:")
    print(f"      Input: 'Create a Git branch'")
    print(f"      Output: 'git checkout -b branch-name'")

if __name__ == "__main__":
    analyze_training_data() 
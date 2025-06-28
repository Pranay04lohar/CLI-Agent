# Improving CLI Agent Fine-Tuning with Real Data

## 🔍 **Root Cause Analysis**

Your current fine-tuned model isn't working well because:

1. **Training Data Mismatch**: Your Stack Overflow data contains verbose explanations, but you need concise command generation
2. **Prompt Format Mismatch**: The model was trained on "Question:" -> "Answer:" format with detailed explanations
3. **Task Complexity**: Converting explanation-heavy content to direct commands is hard

## 🎯 **Solutions Using Real Public Data**

### **Option 1: Better Data Curation from Stack Overflow**

Instead of taking all Q&A pairs, be more selective:

````python
def filter_command_focused_qa(qa_pairs):
    """Filter for Q&A pairs that contain direct commands"""
    filtered = []

    for qa in qa_pairs:
        answer = qa['answer']

        # Only keep answers that contain actual commands
        command_indicators = [
            'git ', '$ ', '```bash', '`git', 'cd ', 'ls ', 'mkdir'
        ]

        if any(indicator in answer for indicator in command_indicators):
            # Extract just the command portions
            commands = extract_commands_from_answer(answer)
            if commands:
                filtered.append({
                    'question': qa['question'],
                    'answer': ' | '.join(commands[:3])  # Max 3 commands
                })

    return filtered
````

### **Option 2: Use GitHub README Command Examples**

Scrape command examples from popular repository READMEs:

```python
# Target repositories with good CLI documentation
repos_to_scrape = [
    'git/git',
    'docker/cli',
    'kubernetes/kubernetes',
    'npm/cli',
    'python/cpython'
]

# Extract installation/usage sections that contain commands
```

### **Option 3: Use Command Documentation**

Extract from official documentation:

- Git official docs
- Docker docs
- Linux man pages in markdown format
- CLI tool documentation

### **Option 4: Curate Your Existing Data Better**

Modify your existing filtering to focus on the command parts:

````python
def extract_essential_commands(stackoverflow_qa):
    """Extract just the essential commands from SO answers"""

    for qa in stackoverflow_qa:
        answer = qa['answer']

        # Extract commands using regex
        git_commands = re.findall(r'git [^\n]*', answer)
        bash_commands = re.findall(r'\$ ([^\n]*)', answer)
        code_blocks = re.findall(r'```bash\n(.*?)\n```', answer, re.DOTALL)

        all_commands = git_commands + bash_commands + code_blocks

        if all_commands:
            # Create a concise answer with just commands
            qa['answer'] = '\n'.join(all_commands[:3])  # Top 3 commands

    return stackoverflow_qa
````

## 🛠 **Immediate Improvements You Can Make**

### **1. Better Prompt Engineering (Current Fix)**

The updated `agent.py` now:

- Uses simpler prompt format matching your training data
- Better extracts commands from verbose SO-style responses
- Has fallbacks for common tasks

### **2. Post-Processing Training Data**

Run this on your existing data:

```bash
cd data
python extract_commands_only.py command_qa_cleaned.json > command_focused_qa.json
```

### **3. Data Augmentation with Real Examples**

Add more focused examples by:

- Manually curating 50-100 high-quality command examples
- Scraping from `awesome-cli` repositories
- Using command cheat sheets (real ones, not generated)

## 📊 **Testing Your Improvements**

Run the evaluation script to see improvements:

```bash
cd evaluation
python test_agent.py
```

Look for:

- Higher success rate
- More relevant commands generated
- Less verbose outputs

## 🎯 **Expected Results**

With these changes, you should see:

- **Before**: Verbose, unfocused responses
- **After**: Direct, actionable commands

Example:

- **Before**: "To create a new Git branch and switch to it, you need to understand that Git branches are..."
- **After**: "git checkout -b <branch-name>"

## 🔄 **Next Steps**

1. **Test the updated agent** with current changes
2. **Run evaluation** to measure improvement
3. **Consider re-training** with better-curated data if needed
4. **Add more real examples** gradually

Remember: The goal is to work with **real, publicly available data** and improve how you process and use it, not to create synthetic data.

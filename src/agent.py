import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Argument parsing
if len(sys.argv) < 2:
    print("Usage: python src/agent.py \"<your instruction>\"")
    sys.exit(1)
user_instruction = sys.argv[1]

# Paths
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or "microsoft/phi-2"
LORA_ADAPTER_PATH = "../lora_adapter/content/lora_adapter"  # Use main adapter directory
LOG_PATH = "../logs/trace.jsonl"

# Print GPU info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA GPU not available. Running on CPU.")

# Load tokenizer and base model for GPU
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto"
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

print("Response:", end=" ")

# Generate plan
# Use a prompt format that matches your Stack Overflow training data
prompt = f"""Question: {user_instruction}

Answer: """

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate with optimized parameters for command generation
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,  # Higher temperature to get more varied responses like SO
    top_p=0.9,        # Higher top_p for more natural language
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the answer part
if "Answer:" in response:
    plan = response.split("Answer:")[-1].strip()
else:
    plan = response.split(prompt)[-1].strip() if prompt in response else response.strip()

# Extract commands from Stack Overflow style responses
def extract_commands_from_text(text):
    """Extract actual shell commands from Stack Overflow style text"""
    commands = []
    lines = text.split('\n')
    
    # First, look for code blocks (multi-line)
    import re
    
    # Extract bash code blocks
    code_blocks = re.findall(r'```bash\n(.*?)\n```', text, re.DOTALL)
    for block in code_blocks:
        # For simple single-line commands in code blocks
        block_lines = block.strip().split('\n')
        for line in block_lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments
                # Check if it's a simple command (not complex scripts)
                if len(line.split()) <= 10 and not any(keyword in line for keyword in ['while', 'for', 'if', 'function']):
                    commands.append(line)
                    break  # Take first simple command from code block
    
    # If we found commands in code blocks, return those
    if commands:
        return commands
    
    # Otherwise, look line by line
    command_starters = ['git', 'cd', 'ls', 'mkdir', 'touch', 'mv', 'cp', 'rm', 'python', 'python3', 'pip', 'pip3', 'npm', 'node', 'docker', 'curl', 'wget', 'tar', 'grep', 'find', 'sudo', 'apt', 'yum', 'brew', 'head', 'tail', 'cat', 'sort', 'uniq', 'wc', 'awk', 'sed']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for common command patterns in SO answers
        # Direct commands (expanded list)
        if any(line.startswith(cmd + ' ') for cmd in command_starters):
            commands.append(line)
        # Commands in code blocks or after $
        elif line.startswith('$ '):
            commands.append(line[2:])
        elif line.startswith('```') and any(cmd in line for cmd in command_starters):
            cmd = line.replace('```bash', '').replace('```', '').strip()
            if cmd:
                commands.append(cmd)
        # Commands in backticks
        elif line.startswith('`') and line.endswith('`') and len(line) > 2:
            cmd = line[1:-1]
            if any(cmd.startswith(c + ' ') for c in command_starters):
                commands.append(cmd)
        # Look for commands embedded in descriptive text
        elif '`' in line:
            # Extract commands from within backticks in descriptive text
            backtick_commands = re.findall(r'`([^`]+)`', line)
            for cmd in backtick_commands:
                if any(cmd.startswith(c + ' ') for c in command_starters):
                    commands.append(cmd)
    
    return commands

# Extract commands using the new function
commands = extract_commands_from_text(plan)

# If no clear commands found, try to extract from the first meaningful lines
if not commands:
    lines = [line.strip() for line in plan.split('\n') if line.strip()]
    for line in lines[:5]:  # Check first 5 lines
        # Look for git-related keywords and try to extract
        if 'git' in line.lower() and any(word in line.lower() for word in ['branch', 'checkout', 'create']):
            # Try to extract a git command from descriptive text
            if 'checkout -b' in line.lower():
                commands.append('git checkout -b <branch-name>')
                break
            elif 'branch' in line.lower() and 'create' in line.lower():
                commands.append('git checkout -b <branch-name>')
                break
        # Look for virtual environment commands
        elif any(word in line.lower() for word in ['venv', 'virtual', 'environment']) and 'python' in line.lower():
            if 'python3 -m venv' in line.lower() or 'python -m venv' in line.lower():
                commands.append('python3 -m venv <env_name>')
                break
        # Look for pip install commands
        elif 'pip' in line.lower() and 'install' in line.lower():
            if 'requests' in line.lower():
                commands.append('pip install requests')
                break
        # Look for file reading commands
        elif any(word in line.lower() for word in ['first', 'lines', 'head']) and any(word in line.lower() for word in ['file', '.log', '.txt']):
            if 'ten' in line.lower() or '10' in line:
                commands.append('head -n 10 <filename>')
                break

# Output the best command or fallback
if commands:
    # Choose the most relevant command instead of just the first one
    def select_best_command(commands, instruction):
        """Select the most relevant command based on the instruction"""
        
        # For branch creation, prioritize 'checkout -b' over plain 'checkout'
        if any(word in instruction.lower() for word in ['create', 'new']) and 'branch' in instruction.lower():
            for cmd in commands:
                if 'checkout -b' in cmd and not any(word in cmd for word in ['merge', 'delete', 'remove']):
                    return cmd
        
        # For file operations, prioritize specific file commands
        if any(word in instruction.lower() for word in ['first', 'lines', 'head']):
            for cmd in commands:
                if cmd.startswith('head'):
                    return cmd
        
        if any(word in instruction.lower() for word in ['last', 'tail']):
            for cmd in commands:
                if cmd.startswith('tail'):
                    return cmd
        
        # For virtual environment, prioritize venv creation
        if any(word in instruction.lower() for word in ['virtual', 'venv', 'environment']):
            for cmd in commands:
                if 'python' in cmd and 'venv' in cmd:
                    return cmd
        
        # For pip, prioritize install commands
        if 'install' in instruction.lower():
            for cmd in commands:
                if 'pip install' in cmd:
                    return cmd
        
        # Default: return the first command
        return commands[0]
    
    best_command = select_best_command(commands, user_instruction)
    print(f"echo {best_command}")
else:
    # Fallback based on common tasks
    if any(word in user_instruction.lower() for word in ['branch', 'git']):
        best_command = "git checkout -b <branch-name>"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['list', 'files']):
        best_command = "ls -la"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['directory', 'folder', 'mkdir']):
        best_command = "mkdir <directory-name>"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['virtual', 'venv', 'environment']):
        best_command = "python3 -m venv <env_name>"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['pip', 'install', 'package']):
        best_command = "pip install <package_name>"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['docker', 'container']):
        best_command = "docker run <image_name>"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['first', 'lines', 'head']) and any(word in user_instruction.lower() for word in ['file', 'log']):
        if any(word in user_instruction.lower() for word in ['ten', '10']):
            best_command = "head -n 10 <filename>"
            print(f"echo {best_command}")
        else:
            best_command = "head <filename>"
            print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['last', 'tail']) and any(word in user_instruction.lower() for word in ['file', 'log']):
        best_command = "tail <filename>"
        print(f"echo {best_command}")
    elif any(word in user_instruction.lower() for word in ['read', 'view', 'show', 'cat']) and any(word in user_instruction.lower() for word in ['file', 'log']):
        best_command = "cat <filename>"
        print(f"echo {best_command}")
    else:
        best_command = "# Command not recognized"
        print(f"echo {best_command}")

# Logging - log only the selected command, not all extracted commands
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "a", encoding="utf-8") as logf:
    if 'best_command' in locals():
        logf.write(json.dumps({"instruction": user_instruction, "step": best_command}) + "\n")
    else:
        logf.write(json.dumps({"instruction": user_instruction, "step": plan[:100]}) + "\n")
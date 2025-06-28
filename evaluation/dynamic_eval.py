import subprocess
import os

PROMPTS = [
    "Create a new Git branch and switch to it.",
    "Compress the folder reports into reports.tar.gz.",
    "List all Python files in the current directory recursively.",
    "Set up a virtual environment and install requests.",
    "Fetch only the first ten lines of a file named output.log.",
    "Remove all .pyc files but keep the .py files intact",
    "Find all files larger than 100MB in the current directory and its subdirectories, then sort them by size"
]

print("# Dynamic Evaluation Results\n")
print("## Scoring Table\n")
print("| Prompt | Agent Output Quality (0-2) | Comments (optional) |")
print("|--------|----------------------------|---------------------|")
for i in range(1, 8):
    print(f"| {i}      |                            |                     |")
print("\n---\n")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for i, prompt in enumerate(PROMPTS, 1):
    print(f"## Prompt {i}")
    print(f"**Prompt:** {prompt}")
    print("**Agent Output:**\n```")
    try:
        result = subprocess.run(
            ["python", "agent.py", prompt],
            cwd=SRC_DIR,
            capture_output=True, text=True, check=True
        )
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"[Error running agent.py: {e}]")
    print("```")
    print("\n---\n") 
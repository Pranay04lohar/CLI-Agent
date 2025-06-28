import json
import re

with open("data/command_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

command_keywords = [
    'git', 'bash', 'grep', 'tar', 'gzip', 'venv', 'ls', 'cd', 'find', 'chmod', 'chown', 'cp', 'mv', 'rm', 'cat', 'echo', 'export', 'python', 'pip', 'virtualenv', 'source'
]

def is_actionable(answer):
    # Contains code block
    if re.search(r'`[^`]+`', answer):
        return True
    # Contains a shell command line (e.g., $ command ...)
    if re.search(r'(^|\\n)\\s*\\$\\s*\\w+', answer):
        return True
    # Contains a command keyword and is at least 30 characters
    if any(kw in answer.lower() for kw in command_keywords) and len(answer) > 30:
        return True
    # Contains numbered or bulleted steps
    if re.search(r'(^|\\n)\\s*\\d+\\. ', answer) or re.search(r'(^|\\n)\\s*\\- ', answer):
        return True
    return False

cleaned = []
removed = []

for qa in data:
    if is_actionable(qa['answer']):
        cleaned.append(qa)
    else:
        removed.append(qa)

print(f"Kept {len(cleaned)} actionable Q&A pairs.")
print(f"Removed {len(removed)} off-topic Q&A pairs.")

with open("data/command_qa_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

# with open("data/removed_offtopic_qa.json", "w", encoding="utf-8") as f:
#     json.dump(removed, f, indent=2, ensure_ascii=False)
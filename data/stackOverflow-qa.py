from datasets import load_dataset
import pandas as pd
import json
import re

# Load the dataset
ds = load_dataset("Kubermatic/stackoverflow_QAs", split="train")
df = pd.DataFrame(ds)

# Define tags and how many Q&A you want for each
tag_targets = {
    'git': 40,
    'bash': 40,
    'grep': 20,
    'venv': 15,
    'tar': 20,
    'gzip': 15
}

all_qa = []
seen_questions = set()

for tag, n in tag_targets.items():
    # Try to match in tag column
    filtered = df[df['tag'].str.lower().str.contains(tag)]
    # If not enough, try to match in question text as well
    if len(filtered) < n:
        extra = df[df['question'].str.lower().str.contains(tag)]
        filtered = pd.concat([filtered, extra]).drop_duplicates()
    count = 0
    for _, row in filtered.iterrows():
        q = row["question"].strip()
        a = row["answer"].strip()
        if q not in seen_questions and q and a:
            all_qa.append({"question": q, "answer": a})
            seen_questions.add(q)
            count += 1
        if count >= n:
            break

# If total is less than 160, sample more Q&A pairs from the rest of the dataset (not already included)
if len(all_qa) < 160:
    needed = 160 - len(all_qa)
    # Exclude already seen questions
    remaining = df[~df['question'].isin(seen_questions)]
    for _, row in remaining.sample(needed, random_state=42).iterrows():
        q = row["question"].strip()
        a = row["answer"].strip()
        if q and a:
            all_qa.append({"question": q, "answer": a})
        if len(all_qa) >= 160:
            break

print(f"Total Q&A pairs collected: {len(all_qa)}")
with open("data/command_qa.json", "w", encoding="utf-8") as f:
    json.dump(all_qa, f, indent=2, ensure_ascii=False)

# --- Validation Section ---
min_question_len = 10
min_answer_len = 10
command_keywords = ['git', 'bash', 'grep', 'tar', 'gzip', 'venv', 'ls', 'cd', 'command', 'shell', 'terminal']

seen_questions = set()
duplicates = set()
bad_entries = []
off_topic = []

for i, qa in enumerate(all_qa):
    q = qa['question'].strip()
    a = qa['answer'].strip()
    # Check for empty or too short
    if len(q) < min_question_len or len(a) < min_answer_len:
        bad_entries.append({"index": i, "reason": "Too short", "question": q, "answer": a})
    # Check for duplicates
    if q in seen_questions:
        duplicates.add(q)
    seen_questions.add(q)
    # Check for command-line content in answer
    if not any(kw in a.lower() for kw in command_keywords):
        off_topic.append({"index": i, "reason": "No command-line keyword in answer", "question": q, "answer": a})
    # Check for code blocks or shell commands
    if not re.search(r'`[^`]+`|\$\s*\w+', a):
        off_topic.append({"index": i, "reason": "No code block or shell command in answer", "question": q, "answer": a})

print(f"Validation Results:")
print(f"  Duplicates found: {len(duplicates)}")
print(f"  Too short entries: {len(bad_entries)}")
print(f"  Potentially off-topic entries: {len(off_topic)}")

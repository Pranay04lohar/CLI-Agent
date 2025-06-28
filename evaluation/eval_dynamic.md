# Dynamic Evaluation Results

## Scoring Table

| Prompt | Agent Output Quality (0-2) |
| ------ | -------------------------- |
| 1      | 2                          |
| 2      | 2                          |
| 3      | 0                          |
| 4      | 1                          |
| 5      | 0                          |
| 6      | 1                          |
| 7      | 1                          |

---

## Prompt 1

**Prompt:** Create a new Git branch and switch to it.
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo git checkout -b <branch-name>
```

---

## Prompt 2

**Prompt:** Compress the folder reports into reports.tar.gz.
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo $ tar czvf reports.tar.gz reports/
```

---

## Prompt 3

**Prompt:** List all Python files in the current directory recursively.
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo ls -la
```

---

## Prompt 4

**Prompt:** Set up a virtual environment and install requests.
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo python3 -m venv env
```

---

## Prompt 5

**Prompt:** Fetch only the first ten lines of a file named output.log.
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo tail -f output.log | head -n10
```

---

## Prompt 6

**Prompt:** Remove all .pyc files but keep the .py files intact
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo find /usr/local/lib/python2.7 -type f -name '*.pyc' | xargs rm -f
```

---

## Prompt 7

**Prompt:** Find all files larger than 100MB in the current directory and its subdirectories, then sort them by size
**Agent Output:**

```
Using GPU: NVIDIA GeForce GTX 1650
Loading tokenizer and base model...
Loading LoRA adapter...
Response: echo find . -type f -mtime +2 -size 100M -print0 | xargs -0 -I {} sh -c 'ls -l {}"{}" > /dev/null' || exit 1
```

---

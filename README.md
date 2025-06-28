# Demo Video:
https://github.com/user-attachments/assets/e8edb76a-97f3-4e82-bebf-bea18a4dbfed


# CLI Agent: Reproducible Build & Usage Guide

## Overview

This project builds a CLI agent that generates step-by-step command-line plans from natural language instructions, using a fine-tuned LLM (TinyLlama-1.1B-Chat) with LoRA adapters.

---

## 1. Requirements

- Python 3.8+
- pip (Python package manager)
- (Recommended) Linux or WSL/Colab with CUDA GPU for training/inference

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Data Preparation

**a. Download and curate Stack Overflow Q&A:**

```bash
cd data
python stackOverflow-qa.py
```

- This downloads and filters Q&A pairs into `data/command_qa.json`.

**b. Filter for actionable CLI Q&A:**

```bash
python filter_actionable_qa.py
```

- Produces `data/command_qa_cleaned.json` (used for training).

**(Optional) Analyze data:**

```bash
python analyze_training_data.py
```

---

## 3. Model Fine-tuning (LoRA Adapter)

**a. Edit and run the notebook:**

- Open `notebooks/model_fine_tuning.ipynb` in Google Colab (recommended for GPU).
- Follow the cells to:
  - Load data (`data/command_qa_cleaned.json`)
  - Fine-tune TinyLlama-1.1B-Chat with LoRA (QLoRA, PEFT)
  - Save adapter to `lora_adapter/content/lora_adapter/`

**b. Artifacts:**

- LoRA adapter weights: `lora_adapter/content/lora_adapter/adapter_model.safetensors`
- Config: `lora_adapter/content/lora_adapter/adapter_config.json`
- Prompt template: `lora_adapter/content/lora_adapter/chat_template.jinja`

---

## 4. Inference: Run the CLI Agent

**a. Usage:**

```bash
python src/agent.py "Describe your CLI task here"
```

- Example:
  ```bash
  python src/agent.py "Create a new Git branch and switch to it"
  ```

**b. Output:**

- Prints the step-by-step plan and extracts the most relevant shell command.

---

## 5. Evaluation

**a. Static (Automated, ROUGE-L):**

```bash
cd evaluation
python static_eval.py
```

- Compares base vs. LoRA model on benchmark prompts.

**b. Dynamic (Human):**

```bash
python dynamic_eval.py
```

- Prints agent outputs for manual scoring.

**c. Quick test:**

```bash
python test_model_output.py
```

- Shows raw model output for a sample prompt.

**d. Batch test:**

```bash
python evaluation/test_agent.py
```

- Runs a suite of CLI tasks and saves results to `logs/evaluation_results.json`.

---

## 6. Directory Structure

- `src/agent.py` — Main agent script (inference)
- `data/` — Data scripts and Q&A files
- `lora_adapter/` — LoRA adapter weights and config
- `notebooks/model_fine_tuning.ipynb` — Training notebook
- `evaluation/` — Evaluation scripts
- `requirements.txt` — Python dependencies

---

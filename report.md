# CLI Agent Project: One-Page Report

## Overview

This project builds a CLI agent that generates step-by-step command-line plans from natural language instructions, using a fine-tuned LLM with LoRA adapters.

## Data Sources

- **Primary:** Stack Overflow Q&A pairs (via public datasets, e.g., Kubermatic/stackoverflow_QAs on Hugging Face)
- **Selection:** Filtered for actionable CLI topics (git, bash, grep, tar, venv, etc.)
- **Final Dataset:** 150 high-quality Q&A pairs (after cleaning, filtering, and manual curation)

## Model & Hyper-parameters

- **Base Model:** TinyLlama-1.1B-Chat (Hugging Face)
- **Adapter:** LoRA (Low-Rank Adaptation)
- **Key Hyper-parameters:**
  - Epochs: 1
  - Batch size: 8
  - Learning rate: 2e-4
  - LoRA rank: 8
  - Max sequence length: 256
- **Hardware:** Google Colab (T4 GPU, 16GB RAM)

## Training Cost & Time

- **Total training time:** ~10 minutes (1 epochs, 150 samples, T4 GPU)
- **Compute cost:** Free (Google Colab)
- **Model size:** ~1.1B parameters (base) + LoRA adapter (~3MB)

## Evaluation Results

### Static (Automated, ROUGE-L)

| Prompt | Base ROUGE-L | LoRA ROUGE-L |
| ------ | ------------ | ------------ |
| 1      | 0.165        | 0.203        |
| 2      | 0.115        | 0.135        |
| 3      | 0.000        | 0.028        |
| 4      | 0.192        | 0.133        |
| 5      | 0.049        | 0.069        |
| 6      | 0.061        | 0.086        |
| 7      | 0.103        | 0.141        |

- LoRA model outperformed base on most prompts, especially for common CLI tasks.

### Dynamic (Human, 0-2 scale)

| Prompt | Agent Output Quality (0-2) |
| ------ | -------------------------- |
| 1      | 2                          |
| 2      | 2                          |
| 3      | 0                          |
| 4      | 1                          |
| 5      | 0                          |
| 6      | 1                          |
| 7      | 1                          |

- Agent is strong on standard tasks (git, tar), but struggles with recursion, file size, and multi-step logic.

## Two Improvement Ideas

1. **Implementing RAG:**

   - Implement retrieval-augmented generation (RAG) or tool-calling, where the agent can look up relevant documentation snippets before generating or validating its plan.

2. **Integrate Interactive Feedback Loops:**
   - Allow users to provide real-time feedback on the agent’s output
   - Add a feedback collection mechanism (CLI prompt or web UI) after each generated plan. Use this feedback to iteratively fine-tune the model or adjust post-processing rules.

---

**Summary:**
Fine-tuning with LoRA adapters on a small, curated dataset improved the model's ability to generate actionable CLI plans for common tasks. Further gains are likely with more data, better prompts, and improved extraction logic.

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from rouge_score import rouge_scorer

# Prompts for evaluation
PROMPTS = [
    "Create a new Git branch and switch to it.",
    "Compress the folder reports into reports.tar.gz.",
    "List all Python files in the current directory recursively.",
    "Set up a virtual environment and install requests.",
    "Fetch only the first ten lines of a file named output.log.",
    "Remove all .pyc files but keep the .py files intact",
    "Find all files larger than 100MB in the current directory and its subdirectories, then sort them by size"
]

# Reference answers (edit as needed for your gold standard)
REFERENCE_ANSWERS = [
    "1. git checkout -b <branch-name>\n2. git switch <branch-name>",
    "tar -czvf reports.tar.gz reports/",
    "find . -name '*.py'",
    "python3 -m venv venv\nsource venv/bin/activate\npip install requests",
    "head -n 10 output.log",
    "find . -name '*.pyc' -delete",
    "find . -type f -size +100M -exec ls -lh {} + | sort -k 5 -rh"
]

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or "microsoft/phi-2"
LORA_ADAPTER_PATH = "../lora_adapter/content/lora_adapter"

def generate_response(model, tokenizer, prompt, device):
    prompt_text = f"""Question: {prompt}\n\nAnswer: """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the answer part
    if "Answer:" in response:
        plan = response.split("Answer:")[-1].strip()
    else:
        plan = response.split(prompt_text)[-1].strip() if prompt_text in response else response.strip()
    return plan

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else "auto"
    )
    # Load fine-tuned model (with LoRA)
    print("Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    # Prepare ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    results = []
    for i, (prompt, reference) in enumerate(zip(PROMPTS, REFERENCE_ANSWERS)):
        print(f"\nPrompt {i+1}: {prompt}")
        # Base model
        base_plan = generate_response(base_model, tokenizer, prompt, device)
        # Fine-tuned model
        lora_plan = generate_response(lora_model, tokenizer, prompt, device)
        # ROUGE-L scores
        base_score = scorer.score(reference, base_plan)["rougeL"].fmeasure
        lora_score = scorer.score(reference, lora_plan)["rougeL"].fmeasure
        results.append({
            "prompt": prompt,
            "reference": reference,
            "base": base_plan,
            "lora": lora_plan,
            "base_rougeL": base_score,
            "lora_rougeL": lora_score
        })

    # Output markdown table
    print("\n---\n\n# Static Evaluation Results\n")
    for i, r in enumerate(results):
        print(f"## Prompt {i+1}")
        print(f"**Prompt:** {r['prompt']}")
        print(f"**Reference:**\n```")
        print(r['reference'])
        print("```")
        print(f"**Base Model Output:**\n```")
        print(r['base'])
        print("```")
        print(f"**Fine-tuned Model Output:**\n```")
        print(r['lora'])
        print("```")
        print(f"| Model | ROUGE-L |\n|-------|---------|\n| Base  | {r['base_rougeL']:.3f} |\n| LoRA  | {r['lora_rougeL']:.3f} |")
        print("\n---\n")

if __name__ == "__main__":
    main() 
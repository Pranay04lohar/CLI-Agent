#!/usr/bin/env python3
"""
Simple test to show raw model output without any processing
"""

import sys
import os
sys.path.append("src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Paths
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "lora_adapter/content/lora_adapter"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# Test with simple prompt
user_instruction = "list all files which are modified/created in the last 5 minutes"
prompt = f"""Question: {user_instruction}

Answer: """

print(f"Input prompt: '{prompt}'")
print("-" * 50)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Show raw output
print("RAW MODEL OUTPUT:")
print("=" * 50)
print(response)
print("=" * 50)

# Extract just the answer
if "Answer:" in response:
    answer = response.split("Answer:")[-1].strip()
    print(f"EXTRACTED ANSWER: '{answer}'") 
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Paths
BASE_MODEL = "./merged_qwen2p5_3b_dapt"
SFT_ADAPTER = "./sft_output"
MERGED_OUTPUT = "./final_merged_model"

print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

print("🔄 Loading SFT adapter...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)

print("🔄 Merging LoRA weights...")
model = model.merge_and_unload()

print("🔄 Saving merged model...")
os.makedirs(MERGED_OUTPUT, exist_ok=True)
model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)

print("🔄 Copying tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_OUTPUT)

print("✅ Merge completed! Model saved to:", MERGED_OUTPUT)

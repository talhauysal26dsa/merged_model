import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def load_model_8gb(model_path):
    """8GB VRAM için optimize edilmiş model yükleme"""
    
    # Memory optimization settings
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔄 Loading model with 8GB optimizations...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # FP16 for memory efficiency
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  # Memory efficient attention
        max_memory={0: "7GB"},      # Reserve 1GB for system
    )
    
    # Additional memory optimizations
    model.config.use_cache = False  # Disable KV cache during generation
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Memory efficient generation"""
    
    # Clear cache before generation
    torch.cuda.empty_cache()
    
    # Tokenize with truncation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        padding=True
    ).to(model.device)
    
    # Generate with memory optimizations
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Disable cache for memory efficiency
        )
    
    # Decode and clean
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    return response

def chat_interface():
    """Interactive chat interface"""
    
    model_path = "./qwen2p5-3b-dapt-sft-merged"  # Hugging Face'den indirilen model
    print("🚀 Loading model for 8GB VRAM...")
    
    try:
        model, tokenizer = load_model_8gb(model_path)
        print("✅ Model loaded successfully!")
        print("💬 Chat interface ready. Type 'quit' to exit.")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if user_input:
                print("🤖 Assistant: ", end="", flush=True)
                response = generate_response(model, tokenizer, user_input)
                print(response)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try reducing max_memory or using CPU offloading")

if __name__ == "__main__":
    chat_interface()

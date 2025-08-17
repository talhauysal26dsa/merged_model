from huggingface_hub import HfApi, login
import os

# Hugging Face'e giriş yapın
# login()  # Bu satırı çalıştırmadan önce token'ınızı ayarlayın

def upload_model():
    """Model'i Hugging Face Hub'a yükle"""
    
    # Model path
    model_path = "./final_merged_model"
    repo_name = "talhauysal26dsa/qwen2p5-3b-dapt-sft-merged"
    
    print(f"🔄 Uploading model to: {repo_name}")
    
    # API instance
    api = HfApi()
    
    # Repository oluştur (eğer yoksa)
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"⚠️ Repository creation warning: {e}")
    
    # Model dosyalarını yükle
    print("🔄 Uploading model files...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model",
        commit_message="Add Qwen2.5-3B DAPT + SFT merged model"
    )
    
    print(f"✅ Model uploaded successfully!")
    print(f"🌐 Access at: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    print("🚀 Hugging Face Hub Upload Script")
    print("📝 Before running, set your HF token:")
    print("   export HF_TOKEN=your_token_here")
    print("   or use: huggingface-cli login")
    print()
    
    # Token kontrolü
    if not os.environ.get("HF_TOKEN") and not os.path.exists(os.path.expanduser("~/.huggingface/token")):
        print("❌ HF token not found. Please login first:")
        print("   huggingface-cli login")
        exit(1)
    
    upload_model()

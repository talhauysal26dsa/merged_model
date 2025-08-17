# Qwen2.5-3B DAPT + SFT Merged Model

Bu repository, Qwen2.5-3B modelinin DAPT (Domain-Adaptive Pre-Training) ve SFT (Supervised Fine-Tuning) ile eğitilmiş merge edilmiş versiyonunu içerir.

## 🚀 Özellikler

- **Base Model**: Qwen2.5-3B DAPT merge edilmiş
- **Fine-tuning**: 433 training sample ile SFT
- **Optimization**: 8GB VRAM için optimize edilmiş
- **Format**: FP16, memory efficient attention

## 📦 Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/talhauysal26dsa/merged_model.git
cd merged_model

# Gerekli paketleri yükleyin
pip install torch transformers accelerate
```

## 💻 Kullanım

### Model İndirme

Model'i Hugging Face Hub'dan indirin:

```bash
# Model'i indir
git lfs install
git clone https://huggingface.co/talhauysal26dsa/qwen2p5-3b-dapt-sft-merged
```

### 8GB VRAM için Optimize Edilmiş Inference

```python
from inference_8gb_optimized import load_model_8gb, generate_response

# Model'i yükleyin (indirdikten sonra)
model, tokenizer = load_model_8gb("./qwen2p5-3b-dapt-sft-merged")

# Chat yapın
prompt = "Merhaba, nasılsın?"
response = generate_response(model, tokenizer, prompt)
print(response)
```

### Interactive Chat Interface

```bash
python inference_8gb_optimized.py
```

## 🔧 Model Detayları

- **Architecture**: Qwen2.5-3B
- **Training Data**: 433 samples
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Memory Usage**: ~7GB VRAM
- **Precision**: FP16

## 📊 Training Konfigürasyonu

```python
# SFT Training Parameters
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 3e-5
num_train_epochs = 3
max_seq_length = 4096
```

## 🎯 Kullanım Alanları

- Türkçe dil modeli
- Domain-specific fine-tuning
- Resource-constrained environments
- 8GB VRAM sistemler

## 📝 Lisans

Bu model Qwen2.5-3B lisansı altında dağıtılmaktadır.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📞 İletişim

- GitHub: [@talhauysal26dsa](https://github.com/talhauysal26dsa)

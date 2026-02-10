# Vision Transformer for CIFAR-10

Implementation of a Vision Transformer (ViT) from scratch in PyTorch for image classification on the CIFAR-10 dataset.

## 📊 Results
- **Test Accuracy**: 63.4%
- **Model**: Custom ViT (patch size=4, embed_dim=64, depth=6, num_heads=4)
- **Training**: 10 epochs, AdamW, lr=3e-4
- **Hardware**: Trained on GPU

> Note: This is a minimal implementation to demonstrate understanding of ViT architecture. Performance can be improved with data augmentation, longer training, and hyperparameter tuning.

## ▶️ How to Use

### Install dependencies
```bash
poetry install
```

## Run Inference
```
poetry run python predict.py --image cat_test.jpg --model vit_model2.pth
cat
```


## 📁 Project Structure
* `model.py` — ViT implementation
* `train.py` — training loop
* `predict.py` — CLI inference
* `config.py` — hyperparameters
* `utils.py` — helper functions (seed, device)
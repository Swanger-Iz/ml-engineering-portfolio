import os
import torch
import argparse
from PIL import Image
from dataset import transform_val
import model as model_module

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help="Image to predict")
parser.add_argument('--model', type=str, default='efficentnet_b0_docclassifier_final.pth', help="Specify the model you want to use")
args = parser.parse_args()

def load_image(image):
    img = Image.open(image).convert('RGB')
    x = transform_val(img).unsqueeze(0)
    
  # [1, 3, 224, 224]
    # x = x.to(device)
    
    return x

def predict_image(model, label_encoder):
    image = load_image(args.image)
    with torch.inference_mode():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        pred_class = label_encoder.inverse_transform([pred_idx])
        
        print(f'Predicted class: {pred_class}, idx: {pred_idx} | Prob: {probs[0][pred_idx]:.4f}')   # 
    return pred_class, probs
        

if __name__ == '__main__':
    model_path = os.path.join(model_module.default_model_dir, args.model)
    model, le = model_module.upload_model(model_path)
    predict_image(model, le)
    
    
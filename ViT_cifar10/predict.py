import model as mm
import config
import utils
import torch
import argparse
from torchvision.transforms import v2
from PIL import Image

transform_inference = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[.5, .5, .5], std=[.5, .5, .5]
    )
])

# Парсим аргументы
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to input image')
parser.add_argument('--model', type=str, required=False, default='vit_model2.pth', help='Path to model checkpoint')
args = parser.parse_args()


def load_model(model_name):
    model = mm.VisionTransformer(
        config.image_size, config.patch_size, config.channels, config.num_classes, config.embed_dim,
        config.depth, config.num_heads, config.mlp_dim, config.drop_rate).to(utils.device)
    
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()
    return model.to(utils.device)

def load_image(image):
    image = Image.open(image)
    image_tensor = transform_inference(image)
    image_tensor = image_tensor.unsqueeze(0)     # [3, 32, 32] -> [1, 3, 32, 32]
    image_tensor = image_tensor.to(utils.device)
    
    return image_tensor

def predict_image(model):
        with torch.inference_mode():
            logits = model(load_image(args.image))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            return config.cifar10_classes[pred_class]

if __name__ == '__main__':
    model = load_model(args.model)
    result = predict_image(model)
    print(result)


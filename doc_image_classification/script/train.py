import torch
from tqdm import tqdm
import utils
import model

device = utils.device
model, loaded_le = model.load_model()


def train_model(model, loader, optimizer, criterion):
    total_loss, correct_preds = 0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device); targets = targets.to(device)
        
        optimizer.zero_grad()
        
        preds = model(imgs)
        loss = criterion(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        correct_preds += (preds.argmax(dim=1) == targets).sum().item()
        
    return total_loss / len(loader.dataset), correct_preds / len(loader.dataset)    


def evaluate_model(model, loader, criterion):
    total_loss, correct_preds = 0, 0
    torch.inference_mode()
    for imgs, targets in loader:
        imgs = imgs.to(device); targets = targets.to(device)
        
        preds = model(imgs)
        loss = criterion(preds, targets)
        
        total_loss += loss.item() * imgs.size(0)
        correct_preds += (preds.argmax(dim=1) == targets).sum().item()  
        
    
    return total_loss / len(loader.dataset), correct_preds / len(loader.dataset)   
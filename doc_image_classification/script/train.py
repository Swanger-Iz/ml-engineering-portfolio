import torch
from tqdm import tqdm
import utils
import model as model_module
import config
import dataset
import argparse
import os

# Парсим аргументы
parser = argparse.ArgumentParser()
parser.add_argument('--out_m_name', type=str, required=True, help='Output model name')


# parser.add_argument('--model', type=str, default=os.path.join('..', 'models', 'efficentnet_b0_docclassifier_final.pth'), help='Enter a model path')
# parser.add_argument('--train_type', type=int, default=1, help='Train classifier (head) or full model')

args = parser.parse_args()

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

if __name__ == '__main__':
    total_train_acc, total_val_acc = [], []
    total_train_loss, total_val_loss = [], []
    device = utils.device
    
    raw_data = dataset.upload_dataset()
    train_dl, val_dl, test_dl = dataset.split_dataset()
    model, le = model_module.load_model()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = config.criterion

    for e in tqdm(range(config.epochs)):
        train_loss, train_acc = train_model(model, train_dl, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, val_dl, criterion)
        
        total_train_acc.append(train_acc); total_train_loss.append(train_loss)
        total_val_acc.append(val_acc); total_val_loss.append(val_acc)
        
        print(f"Epoch: {e+1}/{config.epochs} | Train accuracy: {train_acc:.4f}%, Train loss: {train_loss:.4f} | Validation accuracy: {val_acc:.4f}%, Validation loss: {val_loss:.4f}")
        
    model_module.save_model(model, args.out_m_name)
    print(__name__, 'done')
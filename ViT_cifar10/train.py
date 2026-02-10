import dataset
import model as model_module
import config
import utils
import torch
import torch.optim as optim
from tqdm.auto import tqdm

model = model_module.VisionTransformer(
    config.image_size, config.patch_size, config.channels, config.num_classes, config.embed_dim,
    config.depth, config.num_heads, config.mlp_dim, config.drop_rate).to(utils.device)

optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate)


def train(model, loader, optimizer, criterion):
    # set the mode of the model into training
    model.train()
    
    total_loss, correct_preds = 0, 0
    
    for batch, targets in loader:
        batch, targets = batch.to(utils.device), targets.to(utils.device)
        optimizer.zero_grad()
        
        # 1. Forward pass (model outputs raw logits)
        out = model(batch)
        
        # 2. Calculate loss (per batch)
        loss = criterion(out, targets)
        
        # 3. Perform backprop
        loss.backward()
        
        # 4. Perform gradient descent
        optimizer.step()
        
        total_loss += loss.item() * batch.size(0)
        correct_preds += (out.argmax(dim=1) == targets).sum().item()
        
    return total_loss / len(loader.dataset), correct_preds / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    total_loss, correct_preds = 0, 0
    
    torch.inference_mode()    # with torch.no_grad()
    for batch, targets in loader:
        batch, targets = batch.to(utils.device), targets.to(utils.device)
        
        out = model(batch)
        loss = config.criterion(out, targets)
        
        total_loss += loss.item() * batch.size(0)
        correct_preds += (out.argmax(dim=1) == targets).sum().item()
        
    return total_loss / len(loader.dataset), correct_preds / len(loader.dataset)
    


### Training the model
total_train_acc, total_test_acc = [], []
total_train_loss, total_test_loss = [], []

# epochs = 10

for epoch in tqdm(range(config.epochs)):
    train_loss, train_acc = train(model, dataset.train_dl, optimizer, config.criterion)
    test_loss, test_acc = evaluate(model, dataset.test_dl)
    
    total_train_acc.append(train_acc); total_train_loss.append(train_loss)
    total_test_acc.append(test_acc); total_test_loss.append(test_loss)
    
    print(f"Epoch: {epoch+1}/{config.epochs} | Train accuracy: {train_acc:.4f}%, Train loss: {train_loss:.4f} | Test accuracy: {test_acc:.4f}%, Test loss: {test_loss:.4f}")
    

print(15 * '-', "End training", 15 * '-')
print("Final results:")
# total_train_acc, total_test_acc, total_train_loss, total_test_loss
print(f"Train accuracy: {total_train_acc}%, Train loss: {total_train_loss} | Test accuracy: {total_test_acc}%, Test loss: {total_test_loss}")
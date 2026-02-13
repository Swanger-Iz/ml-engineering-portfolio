import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision.transforms import v2
import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split


transform_train = v2.Compose([
    v2.ToImage(),    # Конвертируем PIL -> image
    v2.ToDtype(torch.float32, scale=True),   # Масштабируем [0, 255] -> [0, 1]
    v2.Resize((224, 224), antialias=True), 
    v2.RandomAffine(degrees=3, translate=(.05, .05), scale=(.95, 1.05)),    # лёгкий поворот + сдвиг + масштаб
    v2.GaussianNoise(mean=.0, sigma=.01),
    v2.Normalize([.5, .5, .5], [.5, .5, .5])
])

transform_val = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224), antialias=True),
    v2.Normalize([.5, .5, .5], [.5, .5, .5])
])

class doc_ds(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform: img = self.transform(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        return img, label

 
def upload_dataset() -> pd.DataFrame:
    classes = config.classes
    
    tvt_files_df = pd.DataFrame(columns=['path', 'labels'])
    for c in classes:
        class_rpath = os.path.relpath(os.path.join(config.train_path, c))
        
        temp_df = pd.DataFrame(columns=['path', 'labels'])
        temp_df['path'] = os.path.join(class_rpath, pd.Series(os.listdir(os.path.join(config.train_path, c))))
        temp_df['labels'] = c
        
        tvt_files_df = pd.concat((tvt_files_df, temp_df), axis=0, ignore_index=True)
    
    return tvt_files_df
    
    

def split_dataset() -> DataLoader:
    le = LabelEncoder()
    tvt_files_df = upload_dataset()
    
    tvt_files_df['label_id'] = le.fit_transform(tvt_files_df['labels'])
    
    full_doc_ds = doc_ds(tvt_files_df.path.to_list(), tvt_files_df.label_id.to_list(), transform=transform_train)
    train_ds, val_ds, test_ds = random_split(full_doc_ds, [.7, .15, .15])
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True)
    
    return train_dl, val_dl, test_dl
    

    
    

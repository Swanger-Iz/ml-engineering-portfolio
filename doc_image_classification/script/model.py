import timm
import torch
import torch.nn as nn
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import typing

default_model_dir = os.path.join('..', 'models')

def load_model(
    model_name: str = 'efficentnet_b0',
    num_classes: int = 4,
    pretrained: bool = False,
    freeze_backborne: bool = False,
    mode: str = 'eval'
) -> typing.Tuple[nn.Module, LabelEncoder]:
    """
    Создаёт и настраивает модель для задачи классификации документов.
    
    Args:
        model_name: имя архитектуры из timm
        num_classes: количество классов
        pretrained: загружать предобученные веса
        freeze_backbone: заморозить backbone (для stage 1 fine-tuning)
        mode: режим обучения или инференса
    
    Returns:
        torch.nn.Module
    """

    model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(default_model_dir,'efficentnet_b0_docclassifier_final.pth'), weights_only=True))
    if mode == 'eval': model.eval()
    
    loaded_le = joblib.load(os.path.join('..', 'labelencoder.pkl'))
    
    if freeze_backborne:
        for param in model.parameters():
            param.requires_grad(False)
        for param in model.classifier.parameters():
            param.requires_grad(False)
    
    return model, loaded_le

def upload_model(model_path: str):
    """
    Загружает уже обученную модель для инференса.
    
    Args:
        model_path: путь до модели
    Returns:
        torch.nn.Module
    """

    model = timm.create_model(model_name='efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_le = joblib.load(os.path.join('..', 'labelencoder.pkl'))
    
    return model, loaded_le
    
def save_model(model, name, path_to_save=default_model_dir):
    torch.save(model.state_dict(), os.path.join(path_to_save, name))
    
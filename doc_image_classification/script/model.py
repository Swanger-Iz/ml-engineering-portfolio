import timm
import torch
import torch.nn as nn
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import typing

def load_model(
    model_name: str = 'efficientnet_b0',
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
    models_folder = os.path.join('..', 'models')
    
    model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(models_folder,'efficentnet_b0_docclassifier_final.pth'), weights_only=True))
    if mode == 'eval': model.eval()
    
    loaded_le = joblib.load(os.path.join('..', 'labelencoder.pkl'))
    
    if freeze_backborne:
        for param in model.parameters():
            param.requires_grad(False)
        for param in model.classifier.parameters():
            param.requires_grad(False)
    
    return model, loaded_le
    
    
    
    
    
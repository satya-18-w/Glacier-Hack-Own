import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def get_optimizer(model, optimizer_name, learning_rate, weight_decay=1e-4):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name, **kwargs):
    if scheduler_name == 'reduce_lr':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                               patience=5, verbose=True)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    else:
        return None
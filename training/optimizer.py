import torch.optim as optim

def get_optimizer(optimizer_name, model_parameters, lr, weight_decay=1e-4):
    if optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
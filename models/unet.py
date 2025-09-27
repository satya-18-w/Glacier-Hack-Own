import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=config.input_channels,
            classes=1,
            activation=config.activation
        )
    
    def forward(self, x):
        return self.model(x)

class AttentionUNet(nn.Module):
    def __init__(self, config):
        super(AttentionUNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=config.input_channels,
            classes=1,
            activation=config.activation,
            attention_type='scse'
        )
    
    def forward(self, x):
        return self.model(x)

class DeepLabV3(nn.Module):
    def __init__(self, config):
        super(DeepLabV3, self).__init__()
        self.model = smp.DeepLabV3(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=config.input_channels,
            classes=1,
            activation=config.activation
        )
    
    def forward(self, x):
        return self.model(x)

class FPN(nn.Module):
    def __init__(self, config):
        super(FPN, self).__init__()
        self.model = smp.FPN(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=config.input_channels,
            classes=1,
            activation=config.activation
        )
    
    def forward(self, x):
        return self.model(x)
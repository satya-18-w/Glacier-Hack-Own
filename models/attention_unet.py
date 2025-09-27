import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet

class GlacierAttentionUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=5, classes=1):
        super(GlacierAttentionUNet, self).__init__()
        self.model = Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid',
            decoder_attention_type='scse'  # Spatial and Channel Squeeze & Excitation
        )
    
    def forward(self, x):
        return self.model(x)
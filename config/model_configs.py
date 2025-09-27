from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    name: str
    encoder: str
    encoder_weights: str
    activation: str
    input_channels: int = 5

# Available model configurations
MODEL_CONFIGS = {
    'unet': ModelConfig('unet', 'resnet34', 'imagenet', 'sigmoid'),
    'attention_unet': ModelConfig('attention_unet', 'resnet34', 'imagenet', 'sigmoid'),
    'deeplabv3': ModelConfig('deeplabv3', 'resnet50', 'imagenet', 'sigmoid'),
    'fpn': ModelConfig('fpn', 'efficientnet-b0', 'imagenet', 'sigmoid'),
}
from .unet import GlacierUNet
from .attention_unet import GlacierAttentionUNet
import segmentation_models_pytorch as smp

def create_model(model_name, config):
    """Factory function to create different segmentation models"""
    if model_name == "unet":
        return GlacierUNet(**config)
    elif model_name == "attention_unet":
        return GlacierAttentionUNet(**config)
    elif model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=config.get('encoder_name', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config.get('in_channels', 5),
            classes=config.get('classes', 1),
            activation='sigmoid'
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
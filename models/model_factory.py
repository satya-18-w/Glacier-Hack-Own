from .unet import UNet, AttentionUNet, DeepLabV3, FPN

class ModelFactory:
    @staticmethod
    def create_model(model_config):
        model_type = model_config.name.lower()
        
        if model_type == 'unet':
            return UNet(model_config)
        elif model_type == 'attention_unet':
            return AttentionUNet(model_config)
        elif model_type == 'deeplabv3':
            return DeepLabV3(model_config)
        elif model_type == 'fpn':
            return FPN(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
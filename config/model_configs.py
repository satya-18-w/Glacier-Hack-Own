MODEL_CONFIGS = {
    "unet": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 5,
        "classes": 1
    },
    "attention_unet": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 5,
        "classes": 1
    },
    "deeplabv3plus": {
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "in_channels": 5,
        "classes": 1
    }
}
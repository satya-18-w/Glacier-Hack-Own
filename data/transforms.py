import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], p=0.8),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.3),
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.0]*5, std=[1.0]*5),
        ToTensorV2(),
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.0]*5, std=[1.0]*5),
        ToTensorV2(),
    ])
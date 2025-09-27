import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=(256, 256)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        ], p=0.3),
        A.Normalize(mean=[0.0]*5, std=[1.0]*5),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=(256, 256)):
    return A.Compose([
        A.Normalize(mean=[0.0]*5, std=[1.0]*5),
        ToTensorV2(),
    ])
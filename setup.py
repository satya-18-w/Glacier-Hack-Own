from setuptools import setup, find_packages

setup(
    name="glacier_segmentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'segmentation-models-pytorch>=0.2.0',
        'albumentations>=1.0.0',
        'optuna>=2.10.0',
        'rasterio>=1.2.0',
        'opencv-python>=4.5.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'numpy>=1.19.0',
    ],
)
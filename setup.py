from setuptools import setup, find_packages

setup(
    name="wikiart_crnn",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "efficientnet-pytorch>=0.7.1",
        "geoopt>=0.5.0",
        "pandas>=1.5.0",
        "Pillow>=9.0.0",
        "numpy>=1.23.0",
    ],
)

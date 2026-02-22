from setuptools import setup, find_packages

setup(
    name="bcNMF",
    version="0.1.0",
    description="Background Contrastive Nonnegative Matrix Factorization",
    author="Yixuan Li, Archer Y. Yang, Yue Li",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22",
        "torch>=2.0",
        "tqdm",
        "scikit-learn",
        "scipy",
        "umap-learn",
        "matplotlib",
        "scanpy",
    ],
)

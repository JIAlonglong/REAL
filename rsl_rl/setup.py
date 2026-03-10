from setuptools import setup, find_packages

setup(
    name='real-rsl-rl',
    version='1.0.0',
    author='REAL Team',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='RL algorithms for REAL (Robust Extreme Agility Learning)',
    python_requires='>=3.8',
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.16.4",
    ],
)

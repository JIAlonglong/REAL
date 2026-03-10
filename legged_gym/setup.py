from setuptools import setup, find_packages

setup(
    name='real-legged-gym',
    version='1.0.0',
    author='REAL Team',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for REAL (Robust Extreme Agility Learning)',
    install_requires=[
        'isaacgym',
        'real-rsl-rl',
        'matplotlib',
    ],
)

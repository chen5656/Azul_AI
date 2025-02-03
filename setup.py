from setuptools import setup, find_packages

setup(
    name="azul_ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium',
        'pygame'
    ]
) 
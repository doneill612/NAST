from setuptools import setup, find_packages

setup(
    name="nast",
    version="0.1",
    author="David O'Neill",
    author_email="doneill612@gmail.com",
    description="Non-Autoregressive Spatial-Temporal Transformer for Time Series forecasting",
    packages=find_packages("src"),
    package_dir={
        '': 'src'
    },
    install_requires=[
        "torch",
        "datasets"
    ],
    zip_safe=False
)
from setuptools import setup
from version import __version__
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='AutoMLpy',
    version=__version__,
    description="This package is an automatic machine learning module whose function"
                " is to optimize the hyper-parameters of an automatic learning model."
                " Code at: https://github.com/JeremieGince/AutoMLpy .",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/JeremieGince/AutoMLpy',
    author='Jérémie Gince',
    author_email='gincejeremie@gmail.com',
    license='Apache 2.0',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "wheel>=0.36.2",
        "tqdm>=4.56.0",
        "numpy>=1.19.5",
        "scikit-learn>=0.24.1",
        "plotly>=4.14.3",
        "pandas>=1.2.1",
        "dash>=1.19.0",
    ],
)

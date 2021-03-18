from setuptools import setup
from .version import __version__

setup(
    name='AutoMLpy',
    version=__version__,
    description="This package is an automatic machine learning module whose function"
                " is to optimize the hyper-parameters of an automatic learning model.",
    url='https://github.com/JeremieGince/AutoMLpy',
    author='Jérémie Gince',
    author_email='gincejeremie@gmail.com',
    license='Apache 2.0',
    packages=['AutoMLpy'],
    zip_safe=False,
)

from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'FjORD Flwr package'
LONG_DESCRIPTION = 'Implementation of FjORD as a flwr baseline'

setup(
    name="fjord",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
)

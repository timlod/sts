""" Setup script for mlp package. """

from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(name="mlp64",
      version="0.0.1",
      author="Benji Kershenbaum, Christoffer Aminoff and Tim Loderhose",
      description=("MLP64 code for audio style transfer."),
      long_description=readme,
      url="https://github.com/timlod/MLP64",
      license=None,
      packages=['mlp64'])

import os
from setuptools import setup
import pathlib


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def read_requirements_file(filename):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(file_path) as f:
        return [line.strip() for line in f]


setup(
    name="gait_in_eight",
    description="gait_in_eight",
    long_description=long_description,
    url="https://github.com/iBims1JFK/gait_in_eight",
    author="Nico Bohlinger, Jonathan Kinzel",
    author_email="nico.bohlinger@gmail.com",
    version="1.0.0",
    packages=["gait_in_eight"],
    install_requires=read_requirements_file("requirements.txt"),
    license="GPLv3",
)

import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    INSTALL_REQUIRES = f.read().splitlines()

about = {}
with open(os.path.join(here, "betfairlightweight", "__version__.py"), "r") as f:
    exec(f.read(), about)

setup(
    name=about["__title__"],
    version=about["__version__"],
    packages=[
        "betfairlightweight",
        "betfairlightweight.endpoints",
        "betfairlightweight.resources",
        "betfairlightweight.streaming",
    ],
    package_dir={"betfairlightweight": "betfairlightweight"},
    install_requires=INSTALL_REQUIRES,
    url=about["__url__"],
    license=about["__license__"],
    author=about["__author__"],
    author_email="",
    description=about["__description__"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
)

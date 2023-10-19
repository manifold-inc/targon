import re
import os
import codecs
import pathlib
from os import path
from io import open
from setuptools import setup, find_packages
from pkg_resources import parse_requirements


def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []
        for req in requirements:
            # For git or other VCS links
            if req.startswith("git+") or "@" in req:
                # check if "egg=" is present in the requirement string
                if "egg=" in req:
                    pkg_name = re.search(r"egg=([a-zA-Z0-9_-]+)", req.strip())
                    if pkg_name:
                        pkg_name = pkg_name.group(1)
                        processed_requirements.append(pkg_name + " @ " + req.strip())
                else:  # handle git links without "egg="
                    # extracting package name from URL assuming it is the last part of the URL before any @ symbol
                    pkg_name = re.search(r"/([a-zA-Z0-9_-]+)(\.git)?(@|$)", req)
                    if pkg_name:
                        pkg_name = pkg_name.group(1)
                        processed_requirements.append(pkg_name + " @ " + req.strip())
            else:
                processed_requirements.append(req)
        return processed_requirements


requirements = read_requirements("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "targon/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="targon",
    version=version_string,
    description="TargonSearchResult is a subnet on bittensor for multi modality inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manifold-inc/targon",
    author="manifold.inc",
    packages=find_packages(),
    include_package_data=True,
    author_email="",
    license="BSL-1.0",
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
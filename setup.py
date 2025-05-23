# Copyright (c) Zirui Zhang

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python

from setuptools import setup

setup(
    name="tactopro",
    version="0.0.1",
    author="Zirui Zhang",
    author_email="zhangzrjerry@outlook.com",
    packages=["tactopro"],
    install_requires=[
    req.strip() for req in open("requirements.txt").readlines() if req.strip() and not req.startswith("#")
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ]
)

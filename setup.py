#!/usr/bin/env python3
"""
Setup script for Academy
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="academy",
    version="0.1.0",
    description="Wissensdestillation für Spezialmodelle (LLM Fine-Tuning)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hans-Jörg Jödike",
    author_email="",  # Add email if desired
    url="https://github.com/your-repo/academy",  # Update with actual repo
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "academy=academy.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",  # Update license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    keywords="llm fine-tuning knowledge-distillation ai",
)
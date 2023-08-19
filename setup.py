from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line]

setup(
    name="tdd-gpt",
    version="0.1",
    description="Autonomous agent for test driven development using GPT-4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gimlet-ai/tdd-gpt-agent",
    author="Rajiv Poddar",
    author_email="rajiv@gimlet.ai",
    license="MIT",
    keywords="tdd-gpt dev-gpt autonomous coding agent web app development gpt-4 langchain codegen",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "tdd-gpt=tdd_gpt.main:main",
        ],
    },
)

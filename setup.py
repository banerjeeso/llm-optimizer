from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-optimizer",
    version="0.1.0",
    author="Your Team",
    description="Reduce LLM API costs by up to 90% with automatic optimizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "anthropic>=0.40.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "google": ["google-generativeai>=0.5.0"],
        "tiktoken": ["tiktoken>=0.7.0"],   # for exact token counts
        "all": [
            "openai>=1.0.0",
            "google-generativeai>=0.5.0",
            "tiktoken>=0.7.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
    keywords="llm anthropic openai cost optimization token caching batch",
)

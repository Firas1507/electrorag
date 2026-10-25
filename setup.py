"""ElectroRAG - Local AI assistant for electronics datasheets."""

from setuptools import setup, find_packages

setup(
    name="electrorag",
    version="1.0.0",
    description="A Retrieval-Augmented Generation system for querying electronics datasheets locally",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ElectroRAG Team",
    author_email="contact@electrorag.dev",
    url="https://github.com/ElectroRAG/ElectroRAG",
    project_urls={
        "Documentation": "https://github.com/ElectroRAG/ElectroRAG/blob/main/USER_GUIDE.md",
    },
    package_data={"electro_rag": ["py.typed"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="RAG retrieval augmented generation electronics datasheets local AI",
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "pypdf>=3.0.0",
        "tiktoken>=0.5.0",
        "ollama>=0.1.0",
        "requests>=2.31.0",
        "pydantic>=2.5.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.8.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
        ],
        "gui": [
            "streamlit>=1.28.0",
            "python-magic-bin>=0.4.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "electrorag=electro_rag.__init__:main",
        ],
    },
    license="MIT",
    platforms=["Windows", "Linux", "macOS"],
)

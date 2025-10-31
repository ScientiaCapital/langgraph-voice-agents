"""
Setup configuration for LangGraph Voice-Enabled Agent Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="langgraph-voice-agents",
    version="0.1.0",
    author="Scientia Capital",
    author_email="contact@scientiacapital.com",
    description="Voice-enabled multi-agent framework built on LangGraph with LiveKit integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ScientiaCapital/langgraph-voice-agents",
    project_urls={
        "Bug Tracker": "https://github.com/ScientiaCapital/langgraph-voice-agents/issues",
        "Documentation": "https://github.com/ScientiaCapital/langgraph-voice-agents#readme",
        "Source Code": "https://github.com/ScientiaCapital/langgraph-voice-agents",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "webrtcvad>=2.0.10",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "structlog>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "langgraph-voice-agents=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

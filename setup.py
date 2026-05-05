from setuptools import setup, find_packages

setup(
    name="lol-draft-recommendation",
    version="0.1.0",
    description="ML-powered League of Legends draft recommendation system",
    author="ryanbtv5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.1.0",
        "fastapi>=0.104.0",
        "pydantic>=2.4.0",
        "pyyaml>=6.0.1",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "ruff>=0.1.0",
        ]
    },
)

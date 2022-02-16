from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="openseries",
    version="0.3.6",
    packages=find_packages(),
    package_data={"openseries": ["*.json"], ".": ["*.json", "coverage.svg"]},
    install_requires=[
        "jsonschema",
        "numpy",
        "pandas",
        "plotly",
        "python-dateutil",
        "python-stdnum",
        "requests",
        "scipy",
        "statsmodels",
        "testfixtures",
    ],
    extras_require={
        "dev": [
            "black",
            "coverage",
            "flake8",
            "flake8-black",
            "nose",
            "nose-timer",
        ],
    },
    url="https://github.com/CaptorAB/OpenSeries",
    license="BSD License",
    author="karrmagadgeteer2",
    author_email="martin.karrin@captor.se",
    description="Package for simple financial time series analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Natural Language :: English",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

from setuptools import find_namespace_packages, setup

setup(
    name="Asreview-FasttextSum",
    version="0.0.1",
    description="ASReivew FasttextSum Feature Extraction extension",
    url="https://github.com/rohitgarud/Asreview-FasttextSum",
    author="Rohit Garud",
    author_email="rohit.garuda1992@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="systematic review, ASReview",
    packages=find_namespace_packages(include=["asreviewcontrib.*"]),
    python_requires="~=3.6",
    install_requires=["asreview>=1.0"],
    entry_points={
        "asreview.models.classifiers": [],
        "asreview.models.feature_extraction": [
            "fasttext_sum = asreviewcontrib.models.fasttextsum:FasttextSum"
        ],
        "asreview.models.balance": [
            # define balance strategy algorithms
        ],
        "asreview.models.query": [
            # define query strategy algorithms
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/rohitgarud/Asreview-FasttextSum/issues",
        "Source": "https://github.com/rohitgarud/Asreview-FasttextSum",
    },
)

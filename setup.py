import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplelinearregression",
    version="0.0.1",
    author="jarnos97",
    author_email="jarno.e.smit@gmail.com",
    description="Audio notification for code execution",
    long_description=long_description,
    url="https://github.com/jarnos97/Simple_Linear_Regression",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'seaborn',
        'statsmodels',
    ]
)

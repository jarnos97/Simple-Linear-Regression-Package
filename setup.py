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
        'cycler>= 0.10.0',
        'kiwisolver>= 1.3.1',
        'matplotlib>= 3.3.3',
        'numpy>=1.19.4',
        'pandas>=1.1.4',
        'patsy>= 0.5.1',
        'Pillow>= 8.0.1',
        'pyparsing>= 2.4.7',
        'python - dateutil>= 2.8.1',
        'pytz>= 2020.4',
        'scipy>= 1.5.4',
        'seaborn>=0.11.0',
        'six>= 1.15.0',
        'statsmodels>= 0.12.1',
    ]
)

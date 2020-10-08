from setuptools import find_packages, setup

# Package meta-data.
NAME = 'scitorch'
DESCRIPTION = 'Leightweight toolbox for numerical experiments in PyTorch'
URL = ''
EMAIL = 'johannesotterbach@gmail.com'
AUTHOR = 'Johannes Otterbach'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = '0.0.1-alpha'


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', 'test')),
    include_package_data=True,
    license='proprietary',
)

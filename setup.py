from setuptools import setup, find_packages

# Define the version directly here instead of importing
__version__ = [line.strip() for line in open('VERSION').readlines()][0]

setup(
    name='targon',
    version=__version__,
    author='Manifold Labs',
    author_email='robert@manifold.inc',
    description='The code for SN4 on bittensor',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://sybil.com',
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines()
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
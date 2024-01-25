from setuptools import setup, find_packages

setup(
    name='targon',  # Replace with your package's name
    version='1.0.0',  # Replace with your package's version
    author='Manifold Labs',  # Replace with your name
    author_email='robert@manifold.inc',  # Replace with your email
    description='The code for SN4 on bittensor',  # Replace with a brief description of your package
    long_description=open('README.md').read(),  # Long description read from the the readme file
    long_description_content_type='text/markdown',  # This is important if your README is in Markdown
    url='http://sybil.com',  # Replace with the URL of your package's website or repository
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[  # List of dependencies, read from requirements.txt
        # Dependencies can be listed here, e.g., 'numpy >= 1.13.3'
        # Or read from a requirements file:
        line.strip() for line in open('requirements.txt').readlines()
    ],
    classifiers=[  # Classifiers help users find your project by categorizing it
        'Programming Language :: Python :: 3',  # Replace "3" with the minimum version your package supports
        'License :: OSI Approved :: MIT License',  # Choose the appropriate license
        'Operating System :: OS Independent',  # Your package's OS compatibility
    ],
    python_requires='>=3.6',  # Minimum version requirement of the Python for your package
)

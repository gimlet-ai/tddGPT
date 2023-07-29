from setuptools import setup, find_packages

setup(
    name='dev-gpt',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your project dependencies here
        # For example: 'numpy>=1.18.0'
    ],
    entry_points={
        'console_scripts': [
            'dev-gpt=dev_gpt.main:main',
        ],
    },
)

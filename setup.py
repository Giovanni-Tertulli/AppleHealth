# setup.py placed at root directory
from setuptools import setup

setup(
    name="activity_summary",
    version='1.0.0',
    author='Giovanni Angelo Tertulli',
    description='This is an example project',
    long_description='This is a longer description for the project',
    keywords='sample, example, setuptools',
    python_requires='>=3.7, <4',
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'jupyter'
    ],
    install_requires=["activity_summary"],
    extras_require={
        'test': ['pytest', 'coverage'],
    },
    package_data={
        'sample': ['ActivitySummary.csv'],
    },
    entry_points={
        'runners': [
            'sample=sample:main',
        ]
    }
)

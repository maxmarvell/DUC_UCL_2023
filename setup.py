from setuptools import setup, find_packages

VERSION = '0.0.1'


setup(
    name='du_quantum_kit',
    version=VERSION,
    author='Max Marvell, Joel Oscar-Mills',
    author_email='max.marvell@hotmail.co.uk',
    description='A package to help with dual unitary calculations',

    packages=find_packages(where='du_quantum_kit'),
    package_dir={'': 'du_quantum_kit'},
    package_data={
        'du_quantum_kit': ['data/*.csv'],
    },
    include_package_data=True,

    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    license='MIT',
    classifiers=[
        'Licence :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],

    install_requires=["autoray>=0.6.7",
                      "cytoolz>=0.12.2",
                      "llvmlite>=0.41.1",
                      "numba>=0.58.1",
                      "numpy>=1.26.2",
                      "opt-einsum>=3.3.0",
                      "pandas>=2.1.3",
                      "psutil>=5.9.6",
                      "python-dateutil>=2.8.2",
                      "pytz>=2023.3.post1",
                      "quimb>=1.6.0",
                      "scipy>=1.11.4",
                      "six>=1.16.0",
                      "toolz>=0.12.0",
                      "tqdm>=4.66.1",
                      "tzdata>=2023.3"],

    python_requires='>=3.10',
)
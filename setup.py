from setuptools import setup, find_packages

VERSION = '0.0.1'


setup(
    name='du_quantum_kit',
    version=VERSION,
    author='Max Marvell, Joel Oscar-Mills',
    author_email='max.marvell@hotmail.co.uk',
    description='Details about the package',
    packages=find_packages(where='du_quantum_kit'),
    package_data={
        'module_name': ['config/rules.yml'],
    },
    include_package_data=True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[],
)
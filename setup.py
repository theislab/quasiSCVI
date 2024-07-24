from setuptools import setup, find_packages

setup(
    name='quasiSCVI',
    version='0.1.0',
    author='Ismail Ben Ayed, Soroor Hediyeh-Zadeh',
    author_email='ismailbenayed5@gmail.com, soroor.hediyehzadeh@helmholtz-munich.de',
    description='An extension of the SCVI model with a QuasiSCVI model.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/theislab/quasiSCVI',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'anndata',
        'pandas',
        'scvi-tools',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
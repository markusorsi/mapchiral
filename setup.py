from setuptools import setup, find_packages

VERSION = '0.0.4' 
DESCRIPTION = 'Chiral MinHashed Atom-Pair Fingerprint  (MAP*) '
LONG_DESCRIPTION = 'Open-source version of the MAP fingerprint, which includes stereochemistry encoding, mapping of hashes to shingles and parallelization.'

# Setting up
setup(
        name="mapchiral", 
        version=VERSION,
        author="Markus Orsi",
        author_email="<markus.orsi@unibe.ch>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], 
        keywords=['cheminformatics', 'mapchiral', 'fingerprint'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
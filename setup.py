# To use a consistent encoding
from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='track',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='A mess of code for tracking moving objects with a telescope',
    long_description=long_description,

    url='https://github.com/bgottula/track',

    author='Brett Gottula',
    author_email='bgottula@gmail.com',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    keywords='astronomy telescopes',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'point>=0.1',
        'ConfigArgParse==0.12.0',
        'ephem>=3.7',
        'inputs>=0.1',
        'numpy',
        'v4l2==0.2.2',
        'requests',
        'MonthDelta>=1.0b',
    ],

    dependency_links=[
        'https://github.com/bgottula/python-v4l2/tarball/master#egg=v4l2-0.2.2',
        'https://github.com/bgottula/point/tarball/master#egg=point-0.1',
    ],

    entry_points={
        'console_scripts':[
            'blind_track = track.blind_track:main',
            'gamepad = track.gamepad_control:main',
            'hybrid_track = track.hybrid_track:main',
            'object_position = track.object_position:main',
            'optical_track = track.optical_track:main',
            'set_when_and_where = track.set_when_and_where:main',
        ],
    },
)

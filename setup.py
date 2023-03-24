# To use a consistent encoding
from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='track',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.4.0',

    description='A mess of code for tracking moving objects with a telescope',
    long_description=long_description,

    url='https://github.com/seeing-things/track',

    author='Brett Gottula',
    author_email='bgottula@gmail.com',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='astronomy telescopes',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    python_requires='>=3.6',

    install_requires=[
        'appdirs>=1.4',
        'asi @ https://github.com/seeing-things/zwo/tarball/master#subdirectory=python',
        'astropy>=4.0',
        'astropy-healpix>=0.4',
        'bs4>=0.0.1',
        'click>=7.0',
        'coloredlogs>=15.0',
        'ConfigArgParse>=1.0',
        'ephem>=3.7',
        'gps>=3.19',
        'imageio>=2.20',
        'influxdb-client>=1.14',  # telemetry database client; requires InfluxDB 2.0
        'inputs>=0.1',
        'lxml',  # parser for bs4
        'matplotlib>=2.1',
        'MonthDelta>=1.0b',
        'numpy',
        # <4.6 have a nasty bug; see https://github.com/seeing-things/track/issues/236
        'opencv-python>=4.6.0.66',
        'ortools',
        'pandas',
        'point @ https://github.com/seeing-things/point/tarball/master',
        'pyftdi>=0.49',  # for laser pointer control
        'requests',
        'scipy',
        'toml',  # parser for InfluxDB CLI configuration files
        'v4l2 @ https://github.com/seeing-things/python-v4l2/tarball/master',  # webcam support
    ],

    setup_requires = [
        'astropy>=4.0',  # so IERS data can be downloaded
    ],

    entry_points={
        'console_scripts':[
            'align = track.align:main',
            'align_guidescope = track.align_guidescope:main',
            'autofocus = track.autofocus:main',
            'fetch_tles = track.fetch_tles:main',
            'gamepad = track.gamepad_control:main',
            'make_offline_ready = track.make_offline_ready:main',
            'object_position = track.object_position:main',
            'skyplot = track.skyplot:main',
            'startracker = track.startracker:main',
            'track = track.__main__:main',
        ],
    },
)

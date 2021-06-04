# To use a consistent encoding
import site
from importlib import reload
from codecs import open
from os import path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def download_iers():
    """Download fresh Astropy IERS data

    Runs the Astropy code that downloads fresh International Earth Rotation and Reference Systems
    (IERS) data, which is how Astropy learns about Earth's polar motion and rotation rate since
    this evolves in an unpredictable manner at the fine-grained level. We probably don't need ultra
    fresh IERS data but it certainly won't hurt to download fresh data during installation. The
    data is downloaded outside of the package itself since track is often run when internet is not
    accessible and we don't want the software to pause in order to download things in the middle of
    execution. It is safer to assume internet is available during installation than during runtime.

    Note that this can't be run before astropy is installed.

    See the following for more info:
    https://docs.astropy.org/en/stable/utils/iers.html
    https://docs.astropy.org/en/stable/api/astropy.utils.iers.IERS_Auto.html
    https://docs.astropy.org/en/stable/api/astropy.utils.iers.LeapSeconds.html
    """
    # reload(site)
    from astropy.utils import iers
    iers.IERS_Auto().open()
    iers.LeapSeconds.auto_open()


class PostDevelopCommand(develop):
    """Download fresh IERS data after installing in develop mode"""
    def run(self):
        develop.run(self)
        download_iers()


class PostInstallCommand(install):
    """Download fresh IERS data after installing"""
    def run(self):
        install.run(self)
        download_iers()


setup(
    name='track',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.3.0',

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
        'ConfigArgParse==0.12.0',
        'ephem>=3.7',
        'gps>=3.19',
        'influxdb>=5.0.0',  # telemetry database client for InfluxDB 1.0
        'inputs>=0.1',
        'lxml',  # parser for bs4
        'matplotlib>=2.1',
        'MonthDelta>=1.0b',
        'numpy',
        'opencv-python',
        'ortools',
        'pandas',
        'point @ https://github.com/seeing-things/point/tarball/master',
        'pyftdi>=0.49',  # for laser pointer control
        'requests',
        'scipy',
        'v4l2 @ https://github.com/seeing-things/python-v4l2/tarball/master',  # webcam support
    ],

    setup_requires = [
        'astropy>=4.0',  # so IERS data can be downloaded
    ],

    entry_points={
        'console_scripts':[
            'align = track.align:main',
            'gamepad = track.gamepad_control:main',
            'object_position = track.object_position:main',
            'skyplot = track.skyplot:main',
            'startracker = track.startracker:main',
            'track = track.__main__:main',
        ],
    },

    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)

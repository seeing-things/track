
Installation

Computer vision support is provided by OpenCV. To use these features you will need to install OpenCV from your distribution's package manager or from source. You will also need to install the Python bindings. On Ubuntu these are found in the python3-opencv package.

Camera support uses the v4l2capture package which compiles C code. This code #includes the libv4l2.h header file which is part of the libv4l-dev package on Debian Linux distributions. You will need to install this package.

Furthermore, the v4l2capture project has forked. This project requires a fork that is *not* registered in PyPi; it exists here instead: https://github.com/gebart/python-v4l2capture/. To ensure that this package is installed correctly, pass the --process-dependency-links option to pip:

pip3 install --process-dependency-links .

This will also be required to get a patched version of the v4l2 package, since the maintainers of that package are apparently deceased or otherwise incapacitated. The branch containing the patch is available here: https://bazaar.launchpad.net/~jgottula/python-v4l2/fix-for-bug-1664158/revision/31 and has also been forked and uploaded here: https://github.com/bgottula/python-v4l2
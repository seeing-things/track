
Installation

Computer vision support is provided by OpenCV. To use these features you will need to install OpenCV from your distribution's package manager or from source. You will also need to install the Python bindings. On Ubuntu these are found in the python3-opencv package.

Camera support uses the v4l2capture package which compiles C code. This code #includes the libv4l2.h header file which is part of the libv4l-dev package on Debian Linux distributions. You will need to install this package.

Something requires Python.h to compile. You will need to install the python3-dev package to make this header file available.

A patched version of the v4l2 package is required, since it contains a number of bugs and its maintainers are apparently deceased or otherwise incapacitated. A branch containing the patched version is available here: https://github.com/bgottula/python-v4l2

To ensure that the patched version of the v4l2 package mentioned above is installed, pass the --process-dependency-links option to pip:

pip3 install --process-dependency-links .

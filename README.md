# Python Version Support

This project is Python 3 compatible. Most of the code can probably run under Python 2.7 but the maintainers do not perform any testing with Python 2 and make no promises. The point package, on which this project depends, is Python 3 only.

# Installation

## Dependencies
One or more Python packages on which this project depends compile C or C++ code. On Debian-based distributions you can install the build-essential package which includes gcc. You will also need to install the python3-dev package to make the Python.h header file available.

Computer vision support is provided by OpenCV. You will need the OpenCV packages and the Python bindings. On Ubuntu this can be accomplished by installing the python3-opencv package using apt which will take care of installing all of the dependencies including the OpenCV packages themselves. These packages are available in the following PPA: ppa:timsc/opencv-3.4.

Camera support requires libv4l to be installed on your system. On Debian-based Linux distributions this is the libv4l package.

Camera support also requires the python-v4l2 package. A patched version is needed since the official version contains a number of bugs and its maintainers are unresponsive. A branch containing the patched version is located here: https://github.com/bgottula/python-v4l2 (but there is no need to clone this). To ensure that the patched version of the v4l2 package mentioned above is installed, pass the `--process-dependency-links` option to pip when installing. You will see an angry warning from pip indicating that this feature is deprecated. We know. Use it anyway.

### Optional Dependencies
Telemetry logging requires InfluxDB to be installed. On Debian based systems this is the influxdb package. To use telemetry logging you will also need to start the InfluxDB service and create a new database.

## Install Command
For the default set of features, install the track package using the following command:
`sudo pip3 install --process-dependency-links .`

If you want to install with telemetry support, do:
`sudo pip3 install --process-dependency-links .[telemetry]`
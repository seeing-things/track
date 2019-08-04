#!/bin/bash

# Make an AVI video out of some JPG images. First argument is a path containing jpeg images with
# names that follow the format "frame_0001.jpg". The frames must be numbered in ascending order
# starting with 0000. The second argument gives the filename of the output, for example
# "mymovie.avi".
ffmpeg -framerate 15 -i $1/frame_%04d.jpg -codec copy $2

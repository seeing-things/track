#!/bin/bash

# make an AVI video out of some JPG images
ffmpeg -framerate 15 -i %04d.jpg output.mkv -codec copy output.avi
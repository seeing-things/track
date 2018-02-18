#!/bin/sh

v4l2-ctl --device=/dev/video1 --set-ctrl gain_automatic=0
v4l2-ctl --device=/dev/video1 --set-ctrl exposure=3000

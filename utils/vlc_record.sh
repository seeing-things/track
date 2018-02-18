#!/bin/bash

if [ $# -lt 1 ]; then
    echo 1>&2 "$0: not enough arguments"
    exit 2
fi

echo streaming from webcam to $1
cvlc v4l2:///dev/video1 --sout="#transcode{acodec=none}:duplicate{dst=file{dst=$1},dst=display}"

#!/bin/bash

set -x
set +e

v4l2-ctl --list-devices
devs=$(v4l2-ctl --list-devices | grep dev)

for d in $devs; do
  v4l2-ctl --device="$d" --all
  v4l2-ctl --device="$d" --list-formats-ext
done

#
# gst-launch-1.0 v4l2src device=/dev/video3 ! video/x-raw,format=RGB16,width=640,height=480,framerate=30/1 ! videoconvert ! autovideosink
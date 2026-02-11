#!/usr/bin/python3

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
from gi.repository import GdkPixbuf
from gi.repository import Gst

import numpy as np
import argparse
import signal
import os
import random
import json
import subprocess
import re
import os.path
from os import path
import cv2
from PIL import Image
from timeit import default_timer as timer

#init gstreamer
Gst.init(None)
Gst.init_check(None)
#init gtk
Gtk.init(None)
Gtk.init_check(None)

def setup_camera(width=760, height=568, framerate=30):
    """
    Call ST's setup_camera.sh to configure the media pipeline.
    Returns (video_device, camera_caps, dcmipp_sensor, main_postproc) tuple.
    """
    config_camera = f"/usr/local/x-linux-ai/resources/setup_camera.sh {width} {height} {framerate} ribbon"
    x = subprocess.check_output(config_camera, shell=True)
    x = x.decode("utf-8")
    print(x)

    video_device_prev = None
    camera_caps_prev = None
    dcmipp_sensor = None
    main_postproc = None

    for line in x.split("\n"):
        # Use startswith() to match only definition lines (not debug output)
        # Iterate all lines so the last (final) value wins
        if line.startswith("V4L_DEVICE_PREV="):
            video_device_prev = line.split('=', 1)[1]
        if line.startswith("V4L2_CAPS_PREV="):
            # Remove spaces - GStreamer caps syntax requires no spaces after commas
            camera_caps_prev = line.split('=', 1)[1].replace(" ", "")
        if line.startswith("DCMIPP_SENSOR="):
            dcmipp_sensor = line.split('=', 1)[1]
        if line.startswith("MAIN_POSTPROC="):
            main_postproc = line.split('=', 1)[1]

    return video_device_prev, camera_caps_prev, dcmipp_sensor, main_postproc

def main():
    video_device, camera_caps, dcmipp_sensor, main_postproc = setup_camera()
    device = f"/dev/{video_device}"
    caps = camera_caps

    print(f"Using device: {device}, caps: {caps}")

    # Create pipeline
    pipeline = Gst.Pipeline()

    # Source
    v4lsrc = Gst.ElementFactory.make("v4l2src", "source")
    v4lsrc.set_property("device", device)

    # Caps filter
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    gst_caps = Gst.Caps.from_string(caps)
    capsfilter.set_property("caps", gst_caps)

    # Queue
    queue = Gst.ElementFactory.make("queue", "queue")
    queue.set_property("max-size-buffers", 1)
    queue.set_property("leaky", 2)

    # Sink
    sink = Gst.ElementFactory.make("autovideosink", "sink")

    # Add elements
    pipeline.add(v4lsrc)
    pipeline.add(capsfilter)
    pipeline.add(queue)
    pipeline.add(sink)

    # Link
    v4lsrc.link(capsfilter)
    capsfilter.link(queue)
    queue.link(sink)

    # Set state
    pipeline.set_state(Gst.State.PLAYING)

    # Bus
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::error", lambda bus, msg: print(f"Error: {msg.parse_error()}"))
    bus.connect("message::eos", lambda bus, msg: print("EOS"))

    # Run
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()

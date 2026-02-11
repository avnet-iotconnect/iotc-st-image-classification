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


print(setup_camera())

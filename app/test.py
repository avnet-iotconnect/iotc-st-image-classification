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

def setup_camera(width=640, height=480, framerate=30):
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

class TestWidget(Gtk.Box):
    def __init__(self):
        super().__init__()
        self.connect('realize', self._on_realize)

    def _on_realize(self, widget):
        self.camera_pipeline_preview_creation()
        self.nn_pipeline_creation()
        # Delay setting state to allow system to settle
        GLib.timeout_add(2000, self.start_pipelines)

    def start_pipelines(self):
        self.pipeline_preview.set_state(Gst.State.PLAYING)
        self.pipeline_nn.set_state(Gst.State.PLAYING)
        print("Pipelines started")

    def camera_pipeline_preview_creation(self):
        self.pipeline_preview = Gst.Pipeline()

        self.v4lsrc_preview = Gst.ElementFactory.make("v4l2src", "source_prev")
        self.v4lsrc_preview.set_property("device", "/dev/video3")

        caps_prev = "video/x-raw,format=RGB16,width=760,height=568,framerate=30/1"
        camera1caps_prev = Gst.Caps.from_string(caps_prev)
        self.camerafilter_prev = Gst.ElementFactory.make("capsfilter", "filter_preview")
        self.camerafilter_prev.set_property("caps", camera1caps_prev)

        self.queue_prev = Gst.ElementFactory.make("queue", "queue-prev")
        self.queue_prev.set_property("max-size-buffers", 1)
        self.queue_prev.set_property("leaky", 2)

        self.gtkwaylandsink = Gst.ElementFactory.make("gtkwaylandsink", "sink")
        self.pack_start(self.gtkwaylandsink.props.widget, True, True, 0)
        self.gtkwaylandsink.props.widget.show()

        self.fps_disp_sink = Gst.ElementFactory.make("fpsdisplaysink", "fpsmeasure1")
        self.fps_disp_sink.set_property("signal-fps-measurements", True)
        self.fps_disp_sink.set_property("fps-update-interval", 2000)
        self.fps_disp_sink.set_property("text-overlay", False)
        self.fps_disp_sink.set_property("video-sink", self.gtkwaylandsink)

        self.pipeline_preview.add(self.v4lsrc_preview)
        self.pipeline_preview.add(self.camerafilter_prev)
        self.pipeline_preview.add(self.queue_prev)
        self.pipeline_preview.add(self.fps_disp_sink)

        self.v4lsrc_preview.link(self.camerafilter_prev)
        self.camerafilter_prev.link(self.queue_prev)
        self.queue_prev.link(self.fps_disp_sink)

        # Bus
        self.bus_preview = self.pipeline_preview.get_bus()
        self.bus_preview.add_signal_watch()
        self.bus_preview.connect("message::error", lambda bus, msg: print(f"Error: {msg.parse_error()}"))
        self.bus_preview.connect("message::eos", lambda bus, msg: print("EOS"))

    def nn_pipeline_creation(self):
        self.pipeline_nn = Gst.Pipeline()

        self.v4lsrc_nn = Gst.ElementFactory.make("v4l2src", "source_nn")
        self.v4lsrc_nn.set_property("device", "/dev/video2")

        caps_nn = "video/x-raw,format=RGB,width=224,height=224,framerate=30/1"
        camera1caps_nn = Gst.Caps.from_string(caps_nn)
        self.camerafilter_nn = Gst.ElementFactory.make("capsfilter", "filter_nn")
        self.camerafilter_nn.set_property("caps", camera1caps_nn)

        self.queue_nn = Gst.ElementFactory.make("queue", "queue-nn")
        self.queue_nn.set_property("max-size-buffers", 1)
        self.queue_nn.set_property("leaky", 2)

        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("caps", camera1caps_nn)
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)

        self.pipeline_nn.add(self.v4lsrc_nn)
        self.pipeline_nn.add(self.camerafilter_nn)
        self.pipeline_nn.add(self.queue_nn)
        self.pipeline_nn.add(self.appsink)

        self.v4lsrc_nn.link(self.camerafilter_nn)
        self.camerafilter_nn.link(self.queue_nn)
        self.queue_nn.link(self.appsink)

        # Bus
        self.bus_nn = self.pipeline_nn.get_bus()
        self.bus_nn.add_signal_watch()
        self.bus_nn.connect("message::error", lambda bus, msg: print(f"Error: {msg.parse_error()}"))
        self.bus_nn.connect("message::eos", lambda bus, msg: print("EOS"))

def main():
    # Create GTK window
    window = Gtk.Window()
    window.set_title("Camera Test")
    window.set_default_size(800, 600)
    window.connect("destroy", Gtk.main_quit)

    # Create the test widget
    widget = TestWidget()
    window.add(widget)

    window.show_all()

    # Run
    Gtk.main()

if __name__ == "__main__":
    main()

#!/usr/bin/python3

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, GLib, Gst
import subprocess

Gst.init(None)
Gtk.init(None)

# Setup camera for dual pipelines
subprocess.call(["/usr/local/x-linux-ai/resources/setup_camera.sh", "760", "568", "30", "224", "224"])

class TestWidget(Gtk.Box):
    def __init__(self):
        super().__init__()
        self.connect('realize', self._on_realize)

    def _on_realize(self, widget):
        self.camera_pipeline_preview_creation()
        self.nn_pipeline_creation()
        GLib.timeout_add(2000, self.start_pipelines)

    def start_pipelines(self):
        self.pipeline_preview.set_state(Gst.State.PLAYING)
        self.pipeline_nn.set_state(Gst.State.PLAYING)

    def camera_pipeline_preview_creation(self):
        self.pipeline_preview = Gst.Pipeline()
        v4lsrc = Gst.ElementFactory.make("v4l2src", "source_prev")
        v4lsrc.set_property("device", "/dev/video3")
        capsfilter = Gst.ElementFactory.make("capsfilter", "filter_preview")
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB16,width=760,height=568,framerate=30/1"))
        queue = Gst.ElementFactory.make("queue", "queue-prev")
        queue.set_property("max-size-buffers", 1)
        queue.set_property("leaky", 2)
        sink = Gst.ElementFactory.make("gtkwaylandsink", "sink")
        fps_sink = Gst.ElementFactory.make("fpsdisplaysink", "fpssink")
        fps_sink.set_property("video-sink", sink)
        self.pipeline_preview.add(v4lsrc)
        self.pipeline_preview.add(capsfilter)
        self.pipeline_preview.add(queue)
        self.pipeline_preview.add(fps_sink)
        v4lsrc.link(capsfilter)
        capsfilter.link(queue)
        queue.link(fps_sink)
        self.pack_start(sink.props.widget, True, True, 0)
        sink.props.widget.show()
        bus = self.pipeline_preview.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", lambda bus, msg: print(f"Error: {msg.parse_error()}"))

    def nn_pipeline_creation(self):
        self.pipeline_nn = Gst.Pipeline()
        v4lsrc = Gst.ElementFactory.make("v4l2src", "source_nn")
        v4lsrc.set_property("device", "/dev/video2")
        capsfilter = Gst.ElementFactory.make("capsfilter", "filter_nn")
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB,width=224,height=224,framerate=30/1"))
        queue = Gst.ElementFactory.make("queue", "queue-nn")
        queue.set_property("max-size-buffers", 1)
        queue.set_property("leaky", 2)
        appsink = Gst.ElementFactory.make("appsink", "appsink")
        appsink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB,width=224,height=224,framerate=30/1"))
        self.pipeline_nn.add(v4lsrc)
        self.pipeline_nn.add(capsfilter)
        self.pipeline_nn.add(queue)
        self.pipeline_nn.add(appsink)
        v4lsrc.link(capsfilter)
        capsfilter.link(queue)
        queue.link(appsink)

def main():
    window = Gtk.Window()
    window.set_default_size(800, 600)
    window.connect("destroy", Gtk.main_quit)
    widget = TestWidget()
    window.add(widget)
    window.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()

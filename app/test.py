"""
ssh -t root@192.168.38.141 'export GST_DEBUG=2; python3 /home/root/app/test.py 2>&1 | head -100'
(executes for several seconds)
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import subprocess
from PIL import Image

print("Starting test")

Gst.init(None)

# Hardcoded values
nn_input_width = 224
nn_input_height = 224
framerate = 30

def setup_camera():
    config_camera = "/usr/local/x-linux-ai/resources/setup_camera.sh 760 568 30"
    x = subprocess.check_output(config_camera, shell=True)
    x = x.decode("utf-8")
    print(x)
    x = x.split("\n")
    for i in x:
        if "V4L_DEVICE_PREV" in i:
            video_device_prev = i.lstrip('V4L_DEVICE_PREV=')
        if "V4L2_CAPS_PREV" in i:
            camera_caps_prev = i.lstrip('V4L2_CAPS_PREV=').replace(" ", "")
    return video_device_prev, camera_caps_prev

video_device_prev, camera_caps_prev = setup_camera()

# Create pipeline
pipeline = Gst.Pipeline()

# creation of the source v4l2src
v4lsrc1 = Gst.ElementFactory.make("v4l2src", "source")
video_device = "/dev/" + str(video_device_prev)
v4lsrc1.set_property("device", video_device)

#creation of the v4l2src caps
caps = str(camera_caps_prev) + ", framerate=" + str(framerate) + "/1"
print("Camera pipeline configuration :", caps)
camera1caps = Gst.Caps.from_string(caps)
camerafilter1 = Gst.ElementFactory.make("capsfilter", "filter1")
camerafilter1.set_property("caps", camera1caps)

# creation of the videoconvert elements
videoformatconverter1 = Gst.ElementFactory.make("videoconvert", "video_convert1")
videoformatconverter2 = Gst.ElementFactory.make("videoconvert", "video_convert2")

tee = Gst.ElementFactory.make("tee", "tee")

# creation and configuration of the queue elements
queue1 = Gst.ElementFactory.make("queue", "queue-1")
queue2 = Gst.ElementFactory.make("queue", "queue-2")
queue1.set_property("max-size-buffers", 1)
queue1.set_property("leaky", 2)
queue2.set_property("max-size-buffers", 1)
queue2.set_property("leaky", 2)

# creation and configuration of the appsink element
appsink = Gst.ElementFactory.make("appsink", "appsink")
nn_caps = "video/x-raw, format = RGB, width=" + str(nn_input_width) + ",height=" + str(nn_input_height)
nncaps = Gst.Caps.from_string(nn_caps)
appsink.set_property("caps", nncaps)
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)
appsink.set_property("max-buffers", 1)
appsink.set_property("drop", True)

# creation of the fakesink for display
fakesink = Gst.ElementFactory.make("fakesink")

# creation of the video rate and video scale elements
video_rate = Gst.ElementFactory.make("videorate", "video-rate")
video_scale = Gst.ElementFactory.make("videoscale", "video-scale")

# Add all elements to the pipeline
pipeline.add(v4lsrc1)
pipeline.add(camerafilter1)
pipeline.add(videoformatconverter1)
pipeline.add(videoformatconverter2)
pipeline.add(tee)
pipeline.add(queue1)
pipeline.add(queue2)
pipeline.add(appsink)
pipeline.add(fakesink)
pipeline.add(video_rate)
pipeline.add(video_scale)

# linking elements together
v4lsrc1.link(video_rate)
video_rate.link(camerafilter1)
camerafilter1.link(tee)
queue1.link(videoformatconverter1)
videoformatconverter1.link(fakesink)
queue2.link(videoformatconverter2)
videoformatconverter2.link(video_scale)
video_scale.link(appsink)
tee.link(queue1)
tee.link(queue2)

cpt_frame = 0

def new_sample(sink):
    global cpt_frame
    sample = sink.emit("pull-sample")
    if sample:
        print("Got frame", cpt_frame)
        if cpt_frame == 0:
            # Save the image
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width, height = caps.get_structure(0).get_value("width"), caps.get_structure(0).get_value("height")
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                img = Image.frombytes("RGB", (width, height), map_info.data)
                img.save("/tmp/test_image.jpg")
                print("Image saved to /tmp/test_image.jpg")
                buffer.unmap(map_info)
        cpt_frame += 1
        if cpt_frame == 1:
            GLib.idle_add(loop.quit)
    return Gst.FlowReturn.OK

appsink.connect("new-sample", new_sample)

# set pipeline playing mode
pipeline.set_state(Gst.State.PLAYING)

bus = pipeline.get_bus()
bus.add_signal_watch()

def msg_error_cb(bus, message):
    print('error message -> {}'.format(message.parse_error()))

bus.connect('message::error', msg_error_cb)

loop = GLib.MainLoop()
GLib.timeout_add_seconds(10, loop.quit)

loop.run()

print("Loop ended")

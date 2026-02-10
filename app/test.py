import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import subprocess
from PIL import Image

print("Starting test")

Gst.init(None)

# Setup camera
config_camera = "/usr/local/x-linux-ai/resources/setup_camera.sh 760 568 30"
x = subprocess.check_output(config_camera, shell=True)
x = x.decode("utf-8")
print(x)

subprocess.run("v4l2-ctl -d /dev/v4l-subdev3 --set-fmt-video=pixelformat=RGB3,width=760,height=568", shell=True)

# Hardcoded values
main_postproc = "/dev/v4l-subdev3"
isp_file = "/usr/local/demo/bin/dcmipp-isp-ctrl"
cpt_frame = 0
isp_first_config = True

def update_isp_config():
    isp_config_gamma_0 = "v4l2-ctl -d " + main_postproc + " -c gamma_correction=0"
    isp_config_gamma_1 = "v4l2-ctl -d " + main_postproc + " -c gamma_correction=1"
    isp_config_whiteb = isp_file + " -i0 "
    isp_config_autoexposure = isp_file + " -g > /dev/null"

    global isp_first_config
    if isp_first_config:
        subprocess.run(isp_config_gamma_0, shell=True)
        subprocess.run(isp_config_gamma_1, shell=True)
        subprocess.run(isp_config_whiteb, shell=True)
        subprocess.run(isp_config_autoexposure, shell=True)
        isp_first_config = False

    if cpt_frame == 0:
        subprocess.run(isp_config_whiteb, shell=True)
        subprocess.run(isp_config_autoexposure, shell=True)

# Pipeline
src = "v4l2src device=/dev/video7 io-mode=4 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! tee name=t"
pipeline_str = f"{src} ! queue ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224 ! appsink name=sink emit-signals=True sync=true t. ! queue ! textoverlay text='Test' ! videoconvert ! fakesink"

print("Creating pipeline")
pipeline = Gst.parse_launch(pipeline_str)
print("Pipeline created")

appsink = pipeline.get_by_name("sink")
print("Appsink found")

def on_new_sample(sink):
    global cpt_frame
    sample = sink.emit("pull-sample")
    if sample:
        print("Got frame", cpt_frame)
        if cpt_frame == 0:
            update_isp_config()
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

appsink.connect("new-sample", on_new_sample)
print("Connected on_new_sample")

print("Setting pipeline to PLAYING")
pipeline.set_state(Gst.State.PLAYING)

bus = pipeline.get_bus()
bus.add_signal_watch()

def on_error(bus, msg):
    err, debug = msg.parse_error()
    print(f"GStreamer Error: {err}")
    if debug:
        print(f"Debug info: {debug}")

bus.connect("message::error", on_error)
print("Connected on_error")

loop = GLib.MainLoop()
GLib.timeout_add_seconds(10, loop.quit)
print("Starting loop")

loop.run()

print("Loop ended")

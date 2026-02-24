# This example's inference approach is based on "How to run inference using the STAI MPU Python API":
# https://wiki.st.com/stm32mpu/wiki/How_to_run_inference_using_the_STAI_MPU_Python_API
# The GST pipeline approach is (somewhat) -based on the X-Linux-AI image classification examples.
# Note debug GST: GST_DEBUG=v4l2*:5

import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from PIL import Image


from stai_mpu import stai_mpu_network

from argparse import ArgumentParser
import numpy as np

from avnet.iotconnect.sdk.lite import Client, DeviceConfig, Callbacks
from avnet.iotconnect.sdk.lite import __version__ as SDK_VERSION
from avnet.iotconnect.sdk.sdklib.mqtt import C2dOta, C2dAck, C2dCommand
from avnet.iotconnect.sdk.lite.client import KvsClient, AwsCredentialsProvider, S3Client

APP_VERSION="1.0.0"



@dataclass
class ClassificationData:
    """ Custom metadata that can be tied to S3 uploads in IoTConnect UI """
    classification: Optional[str] = field(default=None)
    confidence: Optional[float]  = field(default=None)
    model_name: Optional[str] = field(default=None)
    fps: Optional[int] = field(default=None)

@dataclass
class S3CustomData:
    """ Top level data structure for S3 uploads."""
    cf: ClassificationData = field(default_factory=ClassificationData)

s3_client :Optional[S3Client] = None
s3_upload_triggered = False


# forward declare only for now:
stai_inference: Optional['StAiInference'] = None
iotconnect_client: Optional[Client] = None

Gst.init(None)


@dataclass
class StaiImageClassificationTelemetry:
    sdk_version: str
    version: str # APP_VERSION
    model_name: str # Name of the running model
    fps: int # Processing frames per second
    class1: str # Best class/label detected
    confidence1: float  # Best class/label confidence
    class2: str # Second-best class/label detected
    confidence2: float  # Second-best class/label confidence

stai_ic_telemetry = StaiImageClassificationTelemetry(
    sdk_version=SDK_VERSION,
    version=APP_VERSION,
    model_name="unknown",
    fps=0,
    class1="unknown",
    confidence1=0,
    class2="unknown",
    confidence2=0
)


class StAiInference:
    @staticmethod
    def load_labels(filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def write_or_read_model_file_name(model_file):
        """
        If model_file is not None, it will be read from model_name_txt,
        which would pick it up on next application restart if model_file is none.
        IF model_file is None, it will be read from model_name_txt and returned
        """
        model_name_txt = 'model-name.txt'
        if model_file is None:
            try:
                with open(model_name_txt, 'r') as file:
                    model_file = file.readline().strip()
            except Exception:
                pass
            if model_file is None:
                raise ValueError(f"Unable to read model file name from {model_name_txt}")
        else:
            try:
                with open(model_name_txt, 'w') as file:
                    file.write(model_file)
                    file.write(os.linesep) # just for cleanness
            except Exception:
                print(f"Unable to record model name into {model_name_txt}")
        return model_file

    def __init__(self,
                model_file=None,
                 label_file='ImageNetLabels.txt',
                 ):
        self.lock = threading.Lock()
        self.output_tensor_dtype = None # Set when loaded
        self.num_classes = None # Newer models will be 1001 and load We need to handle this.load() will set it

        self.model_file = model_file
        self.stai_model = self.load(StAiInference.write_or_read_model_file_name(model_file))
        self.labels = StAiInference.load_labels(label_file)

    def load(self, model_file):
        with self.lock:
            # this code section is mainly copied for ST instructions
            self.stai_model = None # if application runs while we are loading. Disable inference until we are done.
            stai_model = stai_mpu_network(model_path=model_file, use_hw_acceleration=True)
            # Read input tensor information
            num_inputs = stai_model.get_num_inputs()
            input_tensor_infos = stai_model.get_input_infos()
            for i in range(0, num_inputs):
                input_tensor_shape = input_tensor_infos[i].get_shape()
                input_tensor_name = input_tensor_infos[i].get_name()
                input_tensor_rank = input_tensor_infos[i].get_rank()
                input_tensor_dtype = input_tensor_infos[i].get_dtype()
                print("**Input node: {} -Input_name:{} -Input_dims:{} - input_type:{} -Input_shape:{}"
                    .format(
                        i,
                        input_tensor_name,
                        input_tensor_rank,
                        input_tensor_dtype,
                        input_tensor_shape)
                )
                if input_tensor_infos[i].get_qtype() == "staticAffine":
                    # Reading the input scale and zero point variables
                    input_tensor_scale = input_tensor_infos[i].get_scale()
                    input_tensor_zp = input_tensor_infos[i].get_zero_point()
                if input_tensor_infos[i].get_qtype() == "dynamicFixedPoint":
                    # Reading the dynamic fixed point position
                    input_tensor_dfp_pos = input_tensor_infos[i].get_fixed_point_pos()

            # Read output tensor information
            num_outputs = stai_model.get_num_outputs()
            output_tensor_infos = stai_model.get_output_infos()
            for i in range(0, num_outputs):
                output_tensor_shape = output_tensor_infos[i].get_shape()
                output_tensor_name = output_tensor_infos[i].get_name()
                output_tensor_rank = output_tensor_infos[i].get_rank()
                output_tensor_dtype = output_tensor_infos[i].get_dtype()
                self.output_tensor_dtype = output_tensor_dtype
                self.num_classes=output_tensor_shape[1]
                print("**Output node: {} -Output_name:{} -Output_dims:{} -  Output_type:{} -Output_shape:{}".format(i, output_tensor_name,
                                                                                                                    output_tensor_rank,
                                                                                                                    output_tensor_dtype,
                                                                                                                    output_tensor_shape))
                if output_tensor_infos[i].get_qtype() == "staticAffine":
                    # Reading the output scale and zero point variables
                    output_tensor_scale = output_tensor_infos[i].get_scale()
                    output_tensor_zp = output_tensor_infos[i].get_zero_point()
                if output_tensor_infos[i].get_qtype() == "dynamicFixedPoint":
                    # Reading the dynamic fixed point position
                    output_tensor_dfp_pos = output_tensor_infos[i].get_fixed_point_pos()

            # Reading input image
            input_width = input_tensor_shape[1]
            input_height = input_tensor_shape[2]
            # input_image = Image.open(args.image).resize((input_width, input_height))
            # input_data = np.expand_dims(input_image, axis=0)

            # stai_model.set_input(0, input_image)
            self.stai_model = stai_model # when invoked externally, make sure these are updated
            self.model_file = model_file
            return stai_model

    def inference(self, input_image):
        with self.lock:
            input_data = np.expand_dims(input_image, axis=0)
            self.stai_model.set_input(0, input_data)

            self.stai_model.run() # run once for warmup

            output_data = self.stai_model.get_output(index=0)
            results = np.squeeze(output_data)
            top_k = results.argsort()[-3:][::-1]

            # newer models will have "canvas" at class index 0 and 1001 total.
            offset = 1 if self.num_classes == 1000 else 0
            for i in top_k:
                # if "non-tfhub" model is passed with 1000 classes
                class_name = self.labels[i + offset]
                if self.output_tensor_dtype == np.uint8:
                    pass # print('# {:08.6f}: {}'.format(float(results[i] / 255.0), class_name), end='')
                else:
                    pass # print('# {:08.6f}: {}'.format(float(results[i]), class_name), end='')
            stai_ic_telemetry.class1 = self.labels[top_k[0] + offset]
            stai_ic_telemetry.class2 = self.labels[top_k[1] + offset]
            stai_ic_telemetry.confidence1 = float(round(results[top_k[0]] * 100, 2))
            stai_ic_telemetry.confidence2 = float(round(results[top_k[1]] * 100, 2))
            # print("")

class CameraPipeline:
    @staticmethod
    def setup_camera(width=760, height=568, framerate=30):
        """
        Call ST's setup_camera.sh to configure the media pipeline.
        Returns (video_device, camera_caps, dcmipp_sensor, main_postproc) tuple.
        """
        config_camera = f"/usr/local/x-linux-ai/resources/setup_camera.sh {width} {height} {framerate} 224 224"
        x = subprocess.check_output(config_camera, shell=True)
        x = x.decode("utf-8")
        print(x)

        video_device_prev = None
        camera_caps_prev = None
        video_device_nn = None
        camera_caps_nn = None
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
            if line.startswith("V4L_DEVICE_NN="):
                video_device_nn = line.split('=', 1)[1]
            if line.startswith("V4L2_CAPS_NN="):
                # Remove spaces - GStreamer caps syntax requires no spaces after commas
                camera_caps_nn = line.split('=', 1)[1].replace(" ", "")
            if line.startswith("DCMIPP_SENSOR="):
                dcmipp_sensor = line.split('=', 1)[1]
            if line.startswith("MAIN_POSTPROC="):
                main_postproc = line.split('=', 1)[1]

        return video_device_prev, camera_caps_prev, video_device_nn, camera_caps_nn, dcmipp_sensor, main_postproc

    def __init__(self, model, use_usb_camera=False, show_window=True):
        self.model = model
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.show_window = show_window
        self.webrtc_queue = queue.Queue(maxsize=1)
        self.pipelines = []

        text_overlay = (
            "textoverlay name=overlay "
            "text=\"Camera Stream\" "
            "valignment=top halignment=left "
            "font-desc=\"Sans, 24\" "
            "shaded-background=true"
        )

        if use_usb_camera:
            # USB: single device, tee into inference + preview/display branches
            src = "v4l2src device=/dev/video7 io-mode=4 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert"
            nn_branch = "queue ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224 ! appsink name=nn_sink emit-signals=True sync=false drop=true"
            preview_branch = "queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=preview_sink emit-signals=True sync=false drop=true"
            if show_window:
                display_branch = f"queue ! {text_overlay} ! videoconvert ! autovideosink sync=false"
                pipeline_str = f"{src} ! tee name=t ! {nn_branch} t. ! {preview_branch} t. ! {display_branch}"
            else:
                pipeline_str = f"{src} ! tee name=t ! {nn_branch} t. ! {preview_branch}"
            self.pipeline_preview = Gst.parse_launch(pipeline_str)
            self.pipeline_nn = self.pipeline_preview  # same pipeline
        else:
            # MIPI: two separate V4L2 devices from setup_camera.sh
            prev_device, prev_caps, nn_device, nn_caps, dcmipp_sensor, main_postproc = CameraPipeline.setup_camera(width=760, height=568, framerate=30)
            print(f"MIPI preview: device={prev_device}, caps={prev_caps}")
            print(f"MIPI NN:      device={nn_device}, caps={nn_caps}")

            # NN pipeline: dedicated device, direct to appsink
            self.pipeline_nn = Gst.parse_launch(
                f"v4l2src device={nn_device} ! {nn_caps} ! "
                "queue max-size-buffers=1 leaky=2 ! "
                "appsink name=nn_sink emit-signals=True sync=false drop=true"
            )

            # Preview pipeline: display + WebRTC
            preview_branch = "queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=preview_sink emit-signals=True sync=false drop=true"
            if show_window:
                display_branch = f"queue ! {text_overlay} ! videoconvert ! autovideosink sync=false"
                self.pipeline_preview = Gst.parse_launch(
                    f"v4l2src device={prev_device} ! {prev_caps} ! "
                    f"tee name=t ! {preview_branch} "
                    f"t. ! {display_branch}"
                )
            else:
                self.pipeline_preview = Gst.parse_launch(
                    f"v4l2src device={prev_device} ! {prev_caps} ! "
                    f"{preview_branch}"
                )

        # Connect appsinks
        self.pipeline_nn.get_by_name("nn_sink").connect("new-sample", self.on_nn_frame)
        self.pipeline_preview.get_by_name("preview_sink").connect("new-sample", self.on_preview_frame)
        self.overlay = self.pipeline_preview.get_by_name("overlay")

        # Start pipelines
        for p in set([self.pipeline_nn, self.pipeline_preview]):
            p.set_state(Gst.State.PLAYING)
            bus = p.get_bus()
            bus.add_signal_watch()
            bus.connect("message::error", self.on_error)
            bus.connect("message::warning", self.on_warning)
            self.pipelines.append(p)

    def on_nn_frame(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width, height = caps.get_structure(0).get_value("width"), caps.get_structure(0).get_value("height")
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        img = Image.frombytes("RGB", (width, height), map_info.data)
        self.model.inference(img)

        self.frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.last_time
        if elapsed >= 1.0:
            fps = int(self.frame_count / elapsed)
            if self.show_window and self.overlay:
                text = "Camera Stream"
                if stai_ic_telemetry.class1 is not None:
                    text = f"{stai_ic_telemetry.class1} {round(stai_ic_telemetry.confidence1)}%"
                self.overlay.set_property("text", f"{text}\nFPS: {fps}")
            stai_ic_telemetry.fps = fps
            self.frame_count = 0
            self.last_time = now
            print(f"FPS: {fps}")

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def on_preview_frame(self, sink):
        global s3_upload_triggered
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width, height = caps.get_structure(0).get_value("width"), caps.get_structure(0).get_value("height")
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data).copy()
        try:
            self.webrtc_queue.put_nowait(frame)
        except queue.Full:
            pass

        if s3_upload_triggered:
            s3_upload_triggered = False
            print("Uploading screencap to S3...")
            img = Image.fromarray(frame)
            img.save("/tmp/output_image.jpg")
            upload_screencap(
                label=stai_ic_telemetry.class1,
                confidence=stai_ic_telemetry.confidence1,
                fps=stai_ic_telemetry.fps,
                local_path="/tmp/output_image.jpg"
            )

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"GStreamer Error: {err}")
        if debug:
            print(f"Debug info: {debug}")

    def on_warning(self, bus, msg):
        warn, debug = msg.parse_warning()
        print(f"GStreamer Warning: {warn}")
        if debug:
            print(f"Debug info: {debug}")

# temp hack function
def parse_credentials(f: Path):
    ret = {}
    with open(f, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('export '):
                var, value = line[len('export ')].split('=', 1)
                ret[var] = value.strip('"').strip("'")
    return ret

    # Example: Access a variable
    print(AWS_DEFAULT_REGION)  # Outputs: us-east-1
def print_credentials(provider: AwsCredentialsProvider):
    """
    Example function to print AWS credentials in a format suitable for setting environment variables
    so that aws cli and similar can be used.
    """
    creds = provider.get_credentials()
    # export for Linux with space so that it doesn't record in shell history when pasted
    command = "set" if sys.platform.startswith('win') else " export"
    print(f'{command} AWS_ACCESS_KEY_ID={creds.access_key_id}')
    print(f'{command} AWS_SECRET_ACCESS_KEY={creds.secret_access_key}')
    print(f'{command} AWS_SESSION_TOKEN="{creds.session_token}"')

def on_command(msg: C2dCommand):
    global iotconnect_client, s3_upload_triggered
    print("Received command", msg.command_name, msg.command_args, msg.ack_id)
    if msg.command_name == "capture":
        print("Capturing image to upload to S3...")
        s3_upload_triggered = True
        if msg.ack_id is not None: # it could be a command without "Acknowledgement Required" flag in the device template
            iotconnect_client.send_command_ack(msg, C2dAck.CMD_SUCCESS_WITH_ACK, "Upload Triggered")
    else:
        print("Command %s not implemented!" % msg.command_name)
        if msg.ack_id is not None: # it could be a command without "Acknowledgement Required" flag in the device template
            iotconnect_client.send_command_ack(msg, C2dAck.CMD_FAILED, "Not Implemented")

def on_ota(msg: C2dOta):
    import urllib.request
    error_msg = None
    iotconnect_client.send_ota_ack(msg, C2dAck.OTA_DOWNLOADING)
    model_file = None
    for url in msg.urls:
        print("Downloading OTA file %s from %s" % (url.file_name, url.url))
        try:
            if url.file_name.endswith(".tflite") or url.file_name.endswith(".nb") or url.file_name.endswith(".onnx"):
                model_file = url.file_name
                # This needs special care:
                # Calling stai_mpu_network(model_path=model_file...)
                # most likely it is not closing the opened file, so if we try to overwrite the file in-process
                # it will trigger a bus error and a core dump
                # We apparently cannot urlretrieve() directly to overwrite the file.
                # We must make a copy in /tmp and overwrite the original file.
                # NOTE: There is an assumption here...
                #   This "detection" is primitive and will not work if the file path
                #   becomes mangled (like path cleanup is applied to use absolute instead of relative)
                #   or if a symlink is used and similar.

                if stai_inference.model_file == model_file:
                    print("Overwriting the original file")
                    time.sleep(0.2)
                    tmp = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
                    tmp.close()
                    urllib.request.urlretrieve(url.url, tmp.name)
                    shutil.copy(tmp.name, url.file_name)  # atomically replaces the model file
                    os.unlink(tmp.name)
                else:
                    # otherwise it should be fine to overwrite whatever the intended file directly and without copy
                    urllib.request.urlretrieve(url.url, url.file_name)
            else:
                urllib.request.urlretrieve(url.url, url.file_name)
                print(f"OTA only downloaded file url.file_name, but if these changes",
                      "need to take effect, you may need to manually restart this process!"
                      )

        except Exception as e:
            print("Encountered download error", e)
            error_msg = "Download error for %s" % url.file_name

    if error_msg is not None:
        iotconnect_client.send_ota_ack(msg, C2dAck.OTA_FAILED, error_msg)
        print('Encountered a download processing error "%s". Not restarting.' % error_msg)  # In hopes that someone pushes a better update
    else:
        if model_file:
            print(f"Loading {model_file}...")
            StAiInference.write_or_read_model_file_name(model_file) # load it upon next restart
            stai_inference.load(model_file=model_file)

        iotconnect_client.send_ota_ack(msg, C2dAck.OTA_DOWNLOAD_DONE)


def upload_screencap(label: str, confidence: float, fps: int, local_path: str = Path(__file__).parent / '/tmp/capture.jpg'):
    print("Account S3 Buckets:")
    print(s3_client.get_buckets())
    print("S3 Credentials:")
    print_credentials(s3_client)
    s3_custom_data = S3CustomData()
    # Upload the file into the default bucket with some custom metadata
    # The "cf" object is special and will be displayed by /IOTCONNECT UI
    s3_custom_data.cf.classification = label
    s3_custom_data.cf.confidence = confidence
    s3_custom_data.cf.model_name = stai_inference.model_file
    s3_custom_data.cf.fps = fps

    iotconnect_client.s3_upload(local_path=local_path, custom_values=asdict(s3_custom_data))

def on_disconnect(reason: str, disconnected_from_server: bool):
    print("Disconnected%s. Reason: %s" % (" from server" if disconnected_from_server else "", reason))



def send_telemetry():
    # Send simple data using a basic dictionary
    if iotconnect_client is not None:
        iotconnect_client.send_telemetry(asdict(stai_ic_telemetry))


def iotconnect_client_init():
    global iotconnect_client, s3_client
    device_config = DeviceConfig.from_iotc_device_config_json_file(
        device_config_json_path="iotcDeviceConfig.json",
        device_cert_path="device-cert.pem",
        device_pkey_path="device-pkey.pem"
    )

    iotconnect_client = Client(
        config=device_config,
        callbacks=Callbacks(
            command_cb=on_command,
            ota_cb=on_ota,
            disconnected_cb=on_disconnect
        )
    )

    s3_client = iotconnect_client.get_s3_client()

    if s3_client is None:
        print("S3 Client is not available. Make sure you enabled File Support in your device template.")
    else:
        print("S3 credentials as environment variables:")
        s3_client.obtain_credentials()
        print_credentials(s3_client)


def iotconnect_application_loop():
    if iotconnect_client is not None:
        if not iotconnect_client.is_connected():
            print('(re)connecting...')
            iotconnect_client.connect()
    if iotconnect_client.is_connected():
        stai_ic_telemetry.model_name =  stai_inference.model_file
        send_telemetry()
    return True


def main():
    global stai_inference
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-file', default=None, help='model to be executed.')
    parser.add_argument('-t', '--reporting-interval', default=2, help='IoTConnect reporting interval in seconds')
    parser.add_argument('-u', '--usb', action='store_true', help='Try use USB camera (typically MJPG on /dev/video7).')
    parser.add_argument('-c', '--channel-arn', required=True, help='KVS signaling channel ARN for WebRTC streaming.')
    args = parser.parse_args()

    stai_inference = StAiInference(args.model_file)
    camera = CameraPipeline(stai_inference, use_usb_camera=args.usb, show_window=True)  # Set to False to disable window

    # Launch WebRTC streaming in a background thread (after camera pipeline is set up)
    from app_webrtc import start_webrtc
    threading.Thread(
        target=start_webrtc,
        args=(
            os.environ['AWS_DEFAULT_REGION'],
            args.channel_arn,
            os.environ['AWS_ACCESS_KEY_ID'],
            os.environ['AWS_SECRET_ACCESS_KEY'],
            os.environ.get('AWS_SESSION_TOKEN'),
            camera.webrtc_queue
        ),
        daemon=True
    ).start()

    loop = GLib.MainLoop()

    try:
        iotconnect_client_init()
        GLib.timeout_add(args.reporting_interval * 1000, iotconnect_application_loop)
    except Exception as ex:
        print("Unable to initialize the IoTConnect client. Error was:", ex)
        print("Proceeding without IoTConnect...")
        return

    try:
        loop.run()
    except KeyboardInterrupt:
        for p in camera.pipelines:
            p.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
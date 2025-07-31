import random
import time
from dataclasses import dataclass, asdict
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
from avnet.iotconnect.sdk.sdklib.mqtt import C2dOta, C2dAck

iotconnect_client: Optional[Client] = None

Gst.init(None)

APP_VERSION="1.0.0"
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

    def __init__(self,
                model_file,
                 label_file='../models/ImageNetLabels.txt',
                 ):
        self.output_tensor_dtype = None # Set when loaded
        self.stai_model = None # Avoid warnings
        self.num_classes = None # Newer models will be 1001 and load We need to handle this.load() will set it
        self.load(model_file)
        self.labels = StAiInference.load_labels(label_file)

    def load(self, model_file):
        # this code section is mainly copied for ST instructions
        self.stai_model = stai_mpu_network(model_path=model_file, use_hw_acceleration=True)
        stai_model = self.stai_model # just to maintain "compatibility" with original copy/paste code
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

    def inference(self, input_image):
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
                print('# {:08.6f}: {}'.format(float(results[i] / 255.0), class_name), end='')
            else:
                print('# {:08.6f}: {}'.format(float(results[i]), class_name), end='')
        stai_ic_telemetry.class1 = self.labels[top_k[0] + offset]
        stai_ic_telemetry.class2 = self.labels[top_k[1] + offset]
        stai_ic_telemetry.confidence1 = round(results[top_k[0]] * 100, 2)
        stai_ic_telemetry.confidence2 = round(results[top_k[1]] * 100, 2)
        print("")


class CameraPipeline:
    def __init__(self, model, show_window=False):
        self.model = model
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.show_window = show_window

        # Base pipeline parts
        src = "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert"

        # Text overlay configuration
        text_overlay = (
            "textoverlay name=overlay "
            "text=\"Camera Stream\" "
            "valignment=top halignment=left "
            "font-desc=\"Sans, 24\" "
            "shaded-background=true ! "
        )

        if show_window:
            # Split stream to both processing and display with text overlay
            self.pipeline = Gst.parse_launch(
                f"{src} ! {text_overlay} tee name=t ! "
                "queue ! video/x-raw,format=RGB ! appsink name=sink emit-signals=True sync=true "
                "t. ! queue ! videoconvert ! autovideosink sync=false"
            )
        else:
            # Processing only with text overlay (though it won't be visible)
            self.pipeline = Gst.parse_launch(
                f"{src} ! {text_overlay} video/x-raw,format=RGB ! appsink name=sink emit-signals=True sync=true"
            )

        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self.on_new_frame)
        self.overlay = self.pipeline.get_by_name("overlay")
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_new_frame(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width, height = caps.get_structure(0).get_value("width"), caps.get_structure(0).get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        # Process frame
        img = Image.frombytes("RGB", (width, height), map_info.data).resize((224, 224))
        self.model.inference(img)
        buffer.unmap(map_info)
        # Update frame counter and calculate FPS
        self.frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.last_time

        if elapsed >= 1.0:  # Update text every second
            fps = int(self.frame_count / elapsed)
            if self.show_window:
                self.overlay.set_property("text", f"Camera Stream\nFPS: {fps}")
            print(f"FPS: {fps}")
            stai_ic_telemetry.fps = fps
            self.frame_count = 0
            self.last_time = now

        return Gst.FlowReturn.OK


def on_ota(msg: C2dOta):
    # We just print the URL. The actual handling of the OTA request would be project specific.
    # See the ota-handling.py for more details.
    print("Received OTA request. File: %s Version: %s URL: %s" % (msg.urls[0].file_name, msg.version, msg.urls[0].url))
    # OTA messages always have ack_id, so it is safe to not check for it before sending the ack
    iotconnect_client.send_ota_ack(msg, C2dAck.OTA_DOWNLOAD_FAILED, "Not implemented")

def on_disconnect(reason: str, disconnected_from_server: bool):
    print("Disconnected%s. Reason: %s" % (" from server" if disconnected_from_server else "", reason))


def send_telemetry():
    # Send simple data using a basic dictionary
    if iotconnect_client is not None:
        iotconnect_client.send_telemetry(asdict(stai_ic_telemetry))


def iotconnect_client_init():
    global iotconnect_client
    device_config = DeviceConfig.from_iotc_device_config_json_file(
        device_config_json_path="iotcDeviceConfig.json",
        device_cert_path="device-cert.pem",
        device_pkey_path="device-pkey.pem"
    )

    iotconnect_client = Client(
        config=device_config,
        callbacks=Callbacks(
            command_cb=None,
            ota_cb=on_ota,
            disconnected_cb=on_disconnect
        )
    )

def iotconnect_application_loop():
    if iotconnect_client is not None:
        if not iotconnect_client.is_connected():
            print('(re)connecting...')
            iotconnect_client.connect()
    if iotconnect_client.is_connected():
        send_telemetry()
    return True


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-file', help='model to be executed.')
    parser.add_argument('-t', '--reporting-interval', default=2, help='IoTConnect reporting interval in seconds')
    args = parser.parse_args()

    model = StAiInference(args.model_file)
    camera = CameraPipeline(model, show_window=True)  # Set to False to disable window
    loop = GLib.MainLoop()

    try:
        iotconnect_client_init()
        stai_ic_telemetry.model_name =  args.model_file
        GLib.timeout_add(args.reporting_interval * 1000, iotconnect_application_loop)
    except Exception as ex:
        print("Unable to initialize the IoTConnect client. Error was:", ex)
        print("Proceeding without IoTConnect...")
        return

    try:
        loop.run()
    except KeyboardInterrupt:
        camera.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
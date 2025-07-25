# qat_helper.py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot

class HubLayerQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer): return []
    def get_activations_and_quantizers(self, layer): return []
    def set_quantize_weights(self, layer, qw): pass
    def set_quantize_activations(self, layer, qa): pass
    def get_output_quantizers(self, layer): return []
    def get_config(self): return {}

MODEL_INPUT_SIZE = (224, 224)


def apply_qat_one_epoch(
    *,
    base_url: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
    npz_path: str = "../data/calibration.npz",
    lr: float = 1e-5,
    batch_size: int = 1,
    epochs: int = 1,
):
    # 1. load images
    npz=np.load(npz_path)
    images = npz["representative_data"]
    labels = npz["classes"]

    # 2. build Functional model (logits only)
    inputs = tf.keras.layers.Input(shape=MODEL_INPUT_SIZE + (3,))
    hub_layer = hub.KerasLayer(base_url, trainable=True)
    annotated = tfmot.quantization.keras.quantize_annotate_layer(
        hub_layer, quantize_config=HubLayerQuantizeConfig()
    )
    logits = annotated(inputs)
    model = tf.keras.Model(inputs, logits)

    # 3. apply fake-quant
    with tfmot.quantization.keras.quantize_scope({
        "HubLayerQuantizeConfig": HubLayerQuantizeConfig,
        "KerasLayer": hub.KerasLayer
    }):
        q_model = tfmot.quantization.keras.quantize_apply(model)

    # 4. one-epoch fine-tune
    q_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    ds = (tf.data.Dataset.from_tensor_slices((images, labels))
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    q_model.fit(ds, epochs=epochs, verbose=0)

    # 5. add Softmax
    final = tf.keras.Sequential([q_model, tf.keras.layers.Softmax()])
    final.build((None,) + MODEL_INPUT_SIZE + (3,))
    return final
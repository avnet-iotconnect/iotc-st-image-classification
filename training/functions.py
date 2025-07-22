import logging
import os
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
logging.getLogger('keras').disabled = True

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from PIL import Image
import random


from classes import IMAGENET2012_CLASSES
# Constants
MODEL_INPUT_SIZE = (224, 224)

def normalized_name(name):
    normalized: str = name.lower().replace(" ", "_")
    return normalized.split(',')[0]

def to_synset_id(name):
    """
    Given a class name (with underscores, lowercase), returns the corresponding ImageNet synset ID.

    Args:
        name (str): The class name (e.g., "man_eating_shark", "grey_whale").
                   Expected to be lowercase with underscores (spaces/dashes converted).

    Returns:
        str: The synset ID (e.g., "n01484850") or None if not found.
    """
    # Normalize the input (lowercase, ensure underscores)
    normalized_name = name.lower().replace("-", "_")

    # Iterate through IMAGENET2012_CLASSES to find a matching synonym
    for synset_id, synonyms in IMAGENET2012_CLASSES.items():
        # Split synonyms into parts and normalize each
        synonym_list = [s.strip().lower().replace(" ", "_").replace("-", "_")
                        for s in synonyms.split(",")]

        # Check if the normalized name matches any synonym
        if normalized_name in synonym_list:
            return synset_id

    return None  # Not found

def to_class_index(synset_id):
    return list(IMAGENET2012_CLASSES.keys()).index(synset_id)

def class_name_from_index(index):
    return list(IMAGENET2012_CLASSES.values())[index]

def make_calibration_dataset(image_dir, num_images=500):
    """Scan image_dir recursively and locate all images, then shuffle and pick 500 ."""
    file_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    # Find all images (case-insensitive)
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in file_extensions:
                image_paths.append(os.path.join(root, f))

    if not image_paths or len(image_paths) == 0:
        raise RuntimeError("No images found!")

    random.seed(134) # so we get the predictable set
    random.shuffle(image_paths)
    random.seed() # revert to true random

    image_paths = image_paths[:num_images]
    if len(image_paths) != num_images:
        raise RuntimeError(f"Expected {num_images}, but found only {len(image_paths)}!")


    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB').resize(MODEL_INPUT_SIZE)
            img_array = np.array(img, dtype=np.float32)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            images.append(img_preprocessed)
        except Exception as ex:
            raise RuntimeError(f"Failed to load {path}! Error was", ex)

    if len(images) < num_images:
        print(f"WARNING: Expected {num_images}, but some failed to load {len(image_paths)}!")

    return np.stack(images, axis=0)


def tensors_to_mobilenetv2_trainds(train_imgs, train_labels):
    # Prepare a training dataset from the preloaded images and their true labels

    # Skip pseudo-labeling if true labels are provided
    train_indexes = train_labels  # Use provided labels directly

    # Prepare the dataset itself
    ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_indexes))

    # Apply data augmentation to create variations
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1, fill_mode="constant"),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.GaussianNoise(0.01),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    def augment(image, label):
        image = data_augmentation(image)
        return image, label

    # Create an augmented dataset by repeating and augmenting the single image
    augmented_dataset = ds.repeat(5)  # Create 20 copies
    augmented_dataset = augmented_dataset.map(augment)

    # Add the original image to ensure it's still there
    dataset = ds.concatenate(augmented_dataset)

    # Some tuneups
    # Batch and prefetch
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def convert_images(test_images):
    images = []
    labels = []
    for img_path, true_category in test_images:
        img = Image.open(img_path).convert('RGB').resize(MODEL_INPUT_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)  # float32 [0,255]
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        images.append(img_preprocessed)

        labels.append(to_class_index(to_synset_id(true_category)))
    return images, labels


def gen_synthetic_representative_dataset():
    """Generates synthetic calibration data with geometric patterns generated with DeepSeek"""
    # Pattern 1: Edge stimuli (stripes)
    for _ in range(15):  # Fewer samples since we simplified patterns
        # Horizontal stripes (alternating bands)
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        img[:, ::10, :, :] = np.random.uniform(-0.7, 0.7)  # Every 10th row
        yield [img]

        # Vertical stripes
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        img[:, :, ::10, :] = np.random.uniform(-0.7, 0.7)  # Every 10th column
        yield [img]

    # Pattern 2: Color blocks (simpler than circles)
    for _ in range(15):
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        # Add 3 random rectangles
        for _ in range(3):
            x1, y1 = np.random.randint(0, 180, 2)
            x2, y2 = x1 + np.random.randint(20, 60), y1 + np.random.randint(20, 60)
            img[:, y1:y2, x1:x2, :] = np.random.uniform(-1, 1, 3)
        yield [img]

    # Pattern 3: Gradient fields
    for _ in range(10):
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        # Linear gradient
        axis = np.random.choice(['x', 'y'])
        if axis == 'x':
            vals = np.linspace(-1, 1, 224)
            img += vals.reshape(1, 1, -1, 1)
        else:
            vals = np.linspace(-1, 1, 224)
            img += vals.reshape(1, -1, 1, 1)
        yield [img]

    # Pattern 4: Pure noise (stress test)
    for _ in range(10):
        yield [np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)]


def create_representative_dataset_gen_function(calibration_npz_file_path):
    def representative_dataset_gen():
        with np.load(calibration_npz_file_path) as data:
            images = data[list(data.keys())[0]] # out key is usually "calibration_images"
            for i in range(len(images)):
                yield [images[i:i+1]]
    return representative_dataset_gen


def inference_tflite(model_content, test_images, skip_names=None):
    """Test quantized TFLite model on test images with same behavior as H5 version"""
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    results = []
    total_time = 0

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    print(f"input_details['quantization']={input_details['quantization']}")
    print(f"output_details['quantization']={output_details['quantization']}")
    if 0==output_scale and 0==output_zero_point:
        output_scale = 1 # weird issue when using float32

    for img_path, true_category in test_images:
        img = Image.open(img_path).convert('RGB').resize(MODEL_INPUT_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch) # This produces float32 in [-1, 1]

        # Apply the quantization formula to convert float32 [-1, 1] to the expected int8/uint8 range
        img_quantized = (img_preprocessed / input_scale + input_zero_point).astype(input_details['dtype'])

        # Timing and inference
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details['index'], img_quantized) # Feed the manually quantized data
        interpreter.invoke()
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        total_time += inference_time

        # Get and dequantize output
        preds_quantized = interpreter.get_tensor(output_details['index'])
        preds = (preds_quantized.astype(np.float32) - output_zero_point) * output_scale

        # Identical postprocessing to H5 version
        # Ensure decode_predictions is imported or fully qualified
        decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
        top_class = normalized_name(decoded_preds[0][1].lower()) # Assuming normalized_name is defined
        confidence = decoded_preds[0][2]
        is_correct = top_class == true_category

        name = os.path.basename(img_path).split('.')[0] if isinstance(img_path, str) else ""
        name = name.split('_')[-1]

        if skip_names is not None and skip_names not in name: # Changed 'not skip_names in name' to 'skip_names not in name' for clarity
            results.append({
                'name': name,
                'true_category': true_category,
                'predicted': top_class,
                'confidence': confidence,
                'correct': is_correct,
                'time_ms': inference_time
            })

        print(f"True: {true_category:20} | Pred: {top_class:20} | Name: {name:10} | Conf: {confidence:.3f} | {'✓' if is_correct else '✗'} | {inference_time:.1f}ms")

    accuracy = sum(r['correct'] for r in results) / len(results) if results else 0
    avg_time = total_time / len(results) if results else 0
    print(f"Accuracy: {accuracy:.2%} | Avg Time: {avg_time:.1f}ms")

    return results, accuracy, avg_time


def inference_h5_model(model, test_images, skip_names=None):
    """Test MobileNetV2 model on test images and return results"""

    results = []
    total_time = 0

    for img_path, true_category in test_images:
        img = Image.open(img_path).convert('RGB').resize(MODEL_INPUT_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)  # float32 [0,255]
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

        start_time = time.perf_counter()
        preds = model.predict(img_preprocessed, verbose=0)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        total_time += inference_time

        decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
        top_class = normalized_name(decoded_preds[0][1].lower())
        confidence = decoded_preds[0][2]

        # Use improved accuracy check
        is_correct = top_class == true_category

        name = os.path.basename(img_path).split('.')[0] if isinstance(img_path, str)  else ""
        name = name.split('_')[-1]
        if skip_names is not None and not skip_names in name:
            results.append({
                'name' :  name,
                'true_category': true_category,
                'predicted': top_class,
                'confidence': confidence,
                'correct': is_correct,
                'time_ms': inference_time
            })

        print(f"True: {true_category:20} | Pred: {top_class:20} | Name: {name:10} | Conf: {confidence:.3f} | {'✓' if is_correct else '✗'} | {inference_time:.1f}ms")

    accuracy = sum(r['correct'] for r in results) / len(results)
    avg_time = total_time / len(results)
    print(f"Accuracy: {accuracy:.2%} | Avg Time: {avg_time:.1f}ms")

    return results, accuracy, avg_time


def convert_to_tflite(model, calibration_data, type='cpu:int8-int8'):
    mapping = {
        'cpu:int8-int8':  {
            'input_type': tf.int8,
            'output_type': tf.int8,
            'per_channel': True,     # Ethos-U and ARM CPUs use per-channel
        },
        'st-npu:uint8-float32': {
            'input_type': tf.uint8,
            'output_type': tf.float32,
            'per_channel': False,    # ST wants per-tensor asymmetric
        },
        'st-npu-pc:uint8-float32': {
            'input_type': tf.uint8,
            'output_type': tf.float32,
            'per_channel': True,  # ST types, but per-channel
        },
    }
    if mapping.get(type) is None:
        raise RuntimeError("Unsupported type for convert_to_tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = mapping[type]['input_type']
    converter.inference_output_type = mapping[type]['output_type']

    if calibration_data is not None:
        converter.representative_dataset = create_representative_dataset_gen_function(calibration_data)
    else:
        converter.representative_dataset = gen_synthetic_representative_dataset

    converter._experimental_disable_per_channel = not mapping[type]['per_channel'] # kinda double negation here

    if not mapping[type]['per_channel'] :
        # Optional: Explicitly allow asymmetric activations (default behavior)
        converter._experimental_full_integer_quantization = True
        converter.use_symmetric_quantization = False  # Not strictly needed (asymmetric is default)
        converter.use_weights_symmetric_quantization = False  # Allow weight zero-point != 0

    if type=='cpu:int8-int8':

        # Ethos compatibility flags that should not hurt CPU-only inference (we would hope)
        # All marked stable in TF 2.18 documentation
        converter.target_spec.supported_types = [tf.int8]
        converter._experimental_allow_all_select_tf_ops = False

        # Ethos-U specific optimizations (recommended by ARM)
        converter.experimental_new_quantizer = True  # Stable since TF 2.10

        # Memory optimization for constrained devices
        converter._experimental_enable_resource_variables = False  # Stable
        ############################################################

        # Optional performance tweaks (tested with Ethos delegates)
        # These are stable but less critical
        converter._experimental_preserve_assert_op = False
        converter.experimental_lower_to_saved_model = True

        # Consider these for Ethos-U !
        # converter._experimental_accelerator = "ethos-u"
        # converter._experimental_additional_quantization_patterns = [
        #     "FullyConnected",
        #     "Conv2D",
        #     "DepthwiseConv2D"
        # ]

    return converter.convert()

def fit_mobilenetv2_old(base_model, train_image_set, validation_image_set, learning_rate=0.0001, epochs=5):
    # Create NEW model instance here so we don't alter the base model
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    model.summary()
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    for layer in model.layers[-2:]:  # Unfreeze last conv blocks + head
        layer.trainable = True

    count = 0
    for layer in model.layers:
        if layer.trainable:
            count+=1
    print("Trainable layers=", count)

    # Fine-tune from this layer onwards
    # fine_tune_at = 100
    #
    # # Freeze all the layers before the `fine_tune_at` layer
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = False

    train_images, train_labels = convert_images(train_image_set)
    train_dataset = tensors_to_mobilenetv2_trainds(np.array(train_images), np.array(train_labels))
    print(np.array(train_images).shape, np.array(train_labels).shape)
    #train_dataset = tf.data.Dataset.from_tensor_slices((np.array(train_images), np.array(train_labels)))
    # train_dataset = train_dataset.batch(4)

    images, labels = convert_images(validation_image_set)
    # print(np.array(images).shape, np.array(labels).shape)
    validation_dataset = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
    validation_dataset = validation_dataset.batch(10)
    #validation_dataset = tensors_to_mobilenetv2_trainds(images, labels)


    # Modify existing final layer
    # last_layer = model.layers[-1]
    # last_layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)

    # # Unfreeze strategy (now affects only the new model)
    # for layer in model.layers[:-20]:  # Unfreeze last conv blocks + head
    #     layer.trainable = True

    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=learning_rate,  # Higher for small datasets
    #    decay_steps=100,
    #    decay_rate=0.9)

    # last_layer = model.layers[-1]
    # last_layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)
    # last_layer.activity_regularizer = None  # Drop L1 to avoid over-constraining

    early_stopping = tf.keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=4,
       restore_best_weights=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(train_dataset, epochs=epochs, verbose=2, validation_data=validation_dataset,
                        # callbacks=[early_stopping]
    )
    return model, history

def fit_mobilenetv2(base_model, train_image_set, validation_image_set, learning_rate=0.0001, epochs=5):
    # Clone the base_model to create an independent instance for modification
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())

    # Get the output of the layer immediately before the original 'predictions' Dense layer.
    # From your summary, this is 'global_average_pooling2d'.
    x = model.get_layer('global_average_pooling2d').output

    # Insert the Dropout layer after global average pooling
    x = tf.keras.layers.Dropout(0.5, name='fine_tune_dropout')(x)

    # Reconnect to a new Dense layer with the same number of units as the original.
    # The original 'predictions' layer will not be directly used in the new model graph.
    num_classes = model.layers[-1].units # Get the original 1000 classes
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions_fine_tuned')(x)

    # Create a *new* Keras Model instance with the modified top.
    # This is necessary to correctly insert a layer into the Functional API graph.
    new_model = tf.keras.Model(inputs=model.input, outputs=output)

    # Transfer weights from the original cloned model to the new model.
    # We copy weights for all layers up to (but not including) the original 'predictions' layer.
    # The new 'predictions_fine_tuned' layer will get its weights copied from the original 'predictions' layer.
    for layer_old, layer_new in zip(model.layers[:-1], new_model.layers[:-1]):
        try:
            layer_new.set_weights(layer_old.get_weights())
        except ValueError:
            pass # Skip layers without trainable weights or with shape mismatches (e.g., InputLayer, ReLU, Pooling)

    # Transfer weights for the final Dense layer
    try:
        new_model.get_layer('predictions_fine_tuned').set_weights(model.get_layer('predictions').get_weights())
    except Exception:
        pass # If original predictions layer name is different or structure causes issues, new layer initializes randomly.

    # Assign the newly created model to 'model' for subsequent operations.
    model = new_model

    # Freeze all layers in the modified model first.
    # This ensures a clean slate for setting trainable status.
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze the specific layers you want to fine-tune.
    # Based on the summary, model.layers[-8:] includes the new Dense, Dropout, GlobalAvgPool,
    # and the last few convolutional blocks (Conv_1, block_16_project etc.) that have trainable parameters.
    for layer in model.layers[-4:]:
        layer.trainable = True

    train_images, train_labels = convert_images(train_image_set)
    train_dataset = tensors_to_mobilenetv2_trainds(np.array(train_images), np.array(train_labels))

    images, labels = convert_images(validation_image_set)
    # print(np.array(images).shape, np.array(labels).shape)
    validation_dataset = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
    validation_dataset = validation_dataset.batch(32)
    #validation_dataset = tensors_to_mobilenetv2_trainds(images, labels)

    early_stopping = tf.keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=10, # Increased patience
       restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    callbacks_list = [early_stopping, reduce_lr]

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(train_dataset, epochs=epochs, verbose=2, validation_data=validation_dataset,
                        callbacks=callbacks_list
    )
    return model, history

def stai_print_model_info(stai_model):
    # Read input tensor information
    num_inputs = stai_model.get_num_inputs()
    input_tensor_infos = stai_model.get_input_infos()
    for i in range(0, num_inputs):
        input_tensor_shape = input_tensor_infos[i].get_shape()
        input_tensor_name = input_tensor_infos[i].get_name()
        input_tensor_rank = input_tensor_infos[i].get_rank()
        input_tensor_dtype = input_tensor_infos[i].get_dtype()
        print("**Input node: {} -Input_name:{} -Input_dims:{} - input_type:{} -Input_shape:{}".format(i, input_tensor_name,
                                                                                                    input_tensor_rank,
                                                                                                    input_tensor_dtype,
                                                                                                    input_tensor_shape))
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


def stai_inference(tflite_model_path, test_images, skip_names=None):
    try:
        from stai_mpu import stai_mpu_network
    except Exception:
        print("Not running on target. stai_mpu not available.")
        return [], 0.0, 0.0
    import time

    print('Loading', tflite_model_path, '...')
    stai_model = stai_mpu_network(model_path=tflite_model_path, use_hw_acceleration=True)

    stai_print_model_info(stai_model)

    input_tensor_infos = stai_model.get_input_infos()
    input_tensor_shape = input_tensor_infos[0].get_shape()
    input_tensor_dtype = input_tensor_infos[0].get_dtype()
    output_tensor_infos = stai_model.get_output_infos()
    output_tensor_dtype = output_tensor_infos[0].get_dtype()

    input_mean = 127.5
    input_std = 127.5

    results = []
    total_time = 0

    for img_path, true_category in test_images:

        img = Image.open(img_path).convert("RGB")

        input_width = input_tensor_shape[1]
        input_height = input_tensor_shape[2]
        img_resized = img.resize((input_width, input_height))
        input_data = np.expand_dims(img_resized, axis=0)

        if input_tensor_dtype == np.float32:
            input_data = (np.float32(input_data) - input_mean) / input_std

        stai_model.set_input(0, input_data)
        stai_model.run()

        start_time = time.perf_counter()
        stai_model.run()
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        total_time += inference_time

        output_data = stai_model.get_output(index=0)
        results_output = np.squeeze(output_data)

        top_idx = np.argmax(results_output)
        if output_tensor_dtype == np.uint8:
            confidence = float(results_output[top_idx] / 255.0)
        else:
            confidence = float(results_output[top_idx])

        top_class = class_name_from_index(top_idx)
        top_class = normalized_name(top_class.lower())

        is_correct = top_class == true_category

        name = os.path.basename(img_path).split('.')[0] if isinstance(img_path, str) else ""
        name = name.split('_')[-1]
        if skip_names is not None and not skip_names in name:
            results.append({
                'name': name,
                'true_category': true_category,
                'predicted': top_class,
                'confidence': confidence,
                'correct': is_correct,
                'time_ms': inference_time
            })

        print(f"True: {true_category:20} | Pred: {top_class:20} | Name: {name:10} | Conf: {confidence:.3f} | {'✓' if is_correct else '✗'} | {inference_time:.1f}ms")

    accuracy = sum(r['correct'] for r in results) / len(results)
    avg_time = total_time / len(results)
    print(f"Accuracy: {accuracy:.2%} | Avg Time: {avg_time:.1f}ms")

    return results, accuracy, avg_time

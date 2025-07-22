import sys

import numpy as np

# this weirdness with imports is just so you can write and execute with PyCharm as well as command line. Add parent directory as "Sources Root".
sys.path.append("..")
sys.path.append("../training")
from training import functions

print(f"Loading images from {sys.argv[1]}...")
images_np_array = functions.make_calibration_dataset(sys.argv[1], num_images=500)

print("Saving to calibration.npz...")
np.savez_compressed("calibration.npz",  representative_data=images_np_array)

print("Saving to calibration-small.npz (100 images)...")
np.savez_compressed("calibration-small.npz",  representative_data=images_np_array[:100])

print("Done.")

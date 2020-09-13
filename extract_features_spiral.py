import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from utils import HDF5DatasetWriter
import random
import progressbar
import os
import PIL
import PIL.Image

print(tf.__version__)
'''
drawings:
---------> spiral
---------------> training
-------------------> healthy
-------------------> parkinson
---------------> testing
-------------------> healthy
-------------------> parkinson
---------> wave
---------------> training
-------------------> healthy
-------------------> parkinson
---------------> testing
-------------------> healthy
-------------------> parkinson
'''

'''
Extract features using ResNet50. Apply to all test and train images. Save in hdf5
'''
# make this args
data_dir = Path(r'D:\Docs\Python_code\ParkinsonsSketch\178338_401677_bundle_archive\drawings')
feature_out_dir = r'Features\spiral_features.hdf5' #args['output']

print('[INFO] loading data...')
spiral_train_images = list(data_dir.glob(r'spiral/training/*/*.png'))
random.shuffle(spiral_train_images) # returns in place
NUM_TRAIN_IMAGES = len(spiral_train_images)
print(f'[INFO] number of training images: {NUM_TRAIN_IMAGES}')
spiral_test_images = list(data_dir.glob(r'spiral/testing/*/*.png'))
random.shuffle(spiral_test_images)
NUM_TEST_IMAGES = len(spiral_test_images)
print(f'[INFO] number of test images: {NUM_TEST_IMAGES}')
imagePaths = spiral_train_images + spiral_test_images
print(f'[INFO] total number of images: {len(imagePaths)}')

labels = [x.parent.stem for x in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top = False)

# 2048 of 7 * 7 = 100352 filters as output of ResNet50 layer before FC
dataset = HDF5DatasetWriter((len(imagePaths), 2048 * 7 * 7),
                            feature_out_dir, dataKey="features",
                            bufSize=1000)
dataset.storeClassLabels(le.classes_)

test_image = load_img(imagePaths[0]) #PIL image instance
print(f'[INFO] Original image shape: {np.array(test_image).shape}')
# fig, axs = plt.subplots()
# plt.imshow(test_image)
# plt.title(le.inverse_transform([labels[0]]))
# plt.show()

widgets = ["Evaluating: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
                               widgets=widgets).start()
bs = 16
# loop in batches
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
    # preprocess each image
    for j, imagePath in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224), interpolation='bilinear')
        image = img_to_array(image)

        # expand dims and subtract mean RGB
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    # extract features
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 100352))

    # then added to dataset
    dataset.add(features, batchLabels)

    pbar.update(i)

dataset.close()
pbar.finish()



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from utils.process_data import *
from utils.process_images import *
from pathlib import Path
import pandas as pd

data_dir = Path(r'D:\Docs\Python_code\ParkinsonsSketch\178338_401677_bundle_archive\drawings')

df_spiral_test_images = pd.DataFrame({'path': list(data_dir.glob(r'spiral/testing/*/*.png'))})
df_spiral_test_images['label'] = df_spiral_test_images['path'].map(lambda x: x.parent.stem)
df_spiral_test_images['path'] = df_spiral_test_images['path'].astype(str)
df_test_images = df_spiral_test_images.sample(frac = 1)
print(pdtabulate(df_test_images.sample(random_state = 42)))

batch_size = len(df_test_images['label'])
img_height = 128 #downsample for memory issue
img_width = 128

# image between 0 and 1 with contrast_stretch
spiral_datagen = ImageDataGenerator(preprocessing_function=contrast_stretch)
spiral_test_generator = spiral_datagen.flow_from_dataframe(df_spiral_test_images,
                                                            x_col='path',
                                                            y_col='label',
                                                            target_size=(img_height, img_width),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode="binary",
                                                            shuffle=False)

spiral_model = load_model(r'models\SpiralNet_98.h5')

spiral_test_generator.reset()
spiral_preds = spiral_model.predict_generator(spiral_test_generator, verbose=1)
spiral_predicted_class_indices = np.where(spiral_preds > 0.5, 1, 0).ravel()
class_names = [key for key in spiral_test_generator.class_indices.keys()]
print(class_names)

y_test_spiral = df_spiral_test_images['label'].to_numpy().ravel()
lb = LabelBinarizer()
y_test_spiral = lb.fit_transform(y_test_spiral).ravel()

print(spiral_predicted_class_indices)
print(y_test_spiral)
print(classification_report(y_test_spiral, spiral_predicted_class_indices,
                            target_names=class_names))

#Wave model
df_wave_test_images = pd.DataFrame({'path': list(data_dir.glob(r'wave/testing/*/*.png'))})
df_wave_test_images['label'] = df_wave_test_images['path'].map(lambda x: x.parent.stem)
df_wave_test_images['path'] = df_wave_test_images['path'].astype(str)
df_wave_test_images = df_wave_test_images.sample(frac = 1)
print(pdtabulate(df_wave_test_images.sample(random_state = 42)))

batch_size = len(df_wave_test_images['label'])
img_height = 128 #downsample for memory issue
img_width = 256

# image between 0 and 1 with contrast_stretch
wave_datagen = ImageDataGenerator(preprocessing_function=contrast_stretch)
wave_test_generator = wave_datagen.flow_from_dataframe(df_wave_test_images,
                                                            x_col='path',
                                                            y_col='label',
                                                            target_size=(img_height, img_width),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode="binary",
                                                            shuffle=False)

wave_model = load_model(r'models\WaveNet_95.h5')

wave_test_generator.reset()
wave_preds = wave_model.predict_generator(wave_test_generator, verbose=1)
wave_predicted_class_indices = np.where(wave_preds > 0.5, 1, 0).ravel()
class_names = [key for key in wave_test_generator.class_indices.keys()]
print(class_names)

y_test_wave = df_wave_test_images['label'].to_numpy().ravel()
lb = LabelBinarizer()
y_test_wave = lb.fit_transform(y_test_wave).ravel()

print(wave_predicted_class_indices)
print(y_test_wave)
print(classification_report(y_test_wave, wave_predicted_class_indices,
                            target_names=class_names))

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, \
    EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from pathlib import Path
from utils.process_images import *
from utils.process_data import *
import matplotlib.pyplot as plt
import random

class WaveNet:
    @staticmethod
    def build(width, height, depth, classes):
        #initialize models, channels last
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # CONV => RELU => CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.2))

        # CONV => RELU => CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.2))

        # CONV => RELU => CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.2))

        # sigmoid classifier
        # FC => RELU => DROPOUT => SIGMOID
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model


# make this args
data_dir = Path(r'D:\Docs\Python_code\ParkinsonsSketch\178338_401677_bundle_archive\drawings')

print('[INFO] loading data...')
df_train_images = pd.DataFrame({'path': list(data_dir.glob(r'wave/training/*/*.png'))})
df_train_images['label'] = df_train_images['path'].map(lambda x: x.parent.stem)
df_train_images['path'] = df_train_images['path'].astype(str)
df_train_images = df_train_images.sample(frac = 1) #shuffle
print(pdtabulate(df_train_images.sample(random_state = 42)))
#print(df_train_images['label'].value_counts())
#print(len(df_train_images.index))


df_test_images = pd.DataFrame({'path': list(data_dir.glob(r'wave/testing/*/*.png'))})
df_test_images['label'] = df_test_images['path'].map(lambda x: x.parent.stem)
df_test_images['path'] = df_test_images['path'].astype(str)
df_test_images = df_test_images.sample(frac = 1)
print(pdtabulate(df_test_images.sample(random_state = 42)))

batch_size = 32
img_height = 128 #downsample for memory issue
img_width = 256
val_split = 0.2
num_train_images = len(df_train_images.index)
num_val_images = int(num_train_images*0.2)
print(num_train_images, num_val_images)

# image between 0 and 1 with contrast_stretch
wave_datagen = ImageDataGenerator(rotation_range=5,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    brightness_range=(0.75,1.25),
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    preprocessing_function=contrast_stretch,
                                    validation_split=val_split)

# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
# will not shuffle before split! Need to shuffle first
wave_train_generator = wave_datagen.flow_from_dataframe(df_train_images,
                                                            x_col='path',
                                                            y_col='label',
                                                            subset="training",
                                                            target_size=(img_height, img_width),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode="binary",
                                                            shuffle=True,
                                                            seed=42)
wave_val_generator = wave_datagen.flow_from_dataframe(df_train_images,
                                                            x_col='path',
                                                            y_col='label',
                                                            subset="validation",
                                                            target_size=(img_height, img_width),
                                                            color_mode="grayscale",
                                                            batch_size=batch_size,
                                                            class_mode="binary",
                                                            shuffle=True,
                                                            seed=42)
class_names = [key for key in wave_train_generator.class_indices.keys()]
print(class_names)

#visualize augmentations
# configure batch size and retrieve one batch of images
for X_batch, y_batch in wave_train_generator:
    # create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(img_height, img_width), cmap=plt.get_cmap('gray'))
        plt.title(class_names[int(y_batch[i])])
        plt.axis('off')
    # show the plot
    plt.show()
    break

print("[INFO] compiling model...")
opt = Adam(1e-3)#Nadam(lr=3e-4)
model = WaveNet.build(width=img_width, height=img_height, depth=1,
                        classes=1)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics = ["accuracy"])

model.summary()

tensorboard_callback = TensorBoard(log_dir=r".\logs\tmp")
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = early_stop = EarlyStopping(monitor='val_loss',patience=16,verbose=1)
checkpointer = ModelCheckpoint(r'models\best_model1.h5', monitor='val_accuracy', verbose=1, save_best_only=True,
                               save_weights_only=False)
callbacks = [tensorboard_callback, reduce_lr, early_stop, checkpointer]

model.fit_generator(wave_train_generator,
                    steps_per_epoch=2000 // batch_size,
                    validation_data=wave_val_generator,
                    validation_steps=400 // batch_size,
                    epochs=100,
                    max_queue_size=10,
                    callbacks=callbacks, verbose=2)

import numpy as np
import tensorflow as tf

from ConvolutionalTransformer import ConTransformer
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau

# image dimesnions, pixel count and color channels
imageSize = 96
channel = 3     # if channel is 1, add grayscale to dataset loader

# loading train subset of dataset
trainImages = keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=554,
    image_size=(imageSize, imageSize),
    batch_size=32,
    )  

# loading validation subset of dataset
valImages = keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=554,
    image_size=(imageSize, imageSize),
    batch_size=32,
    )

# normalizing images
def normalize_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

trainImages = trainImages.map(normalize_images)
valImages = valImages.map(normalize_images)

# weighting classes
labels = []
for _, label in trainImages:
    labels.extend(label.numpy())
labels = np.array(labels)
classWeights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
weights = {i: classWeights[i] for i in range(len(classWeights))}


# model
model = keras.Sequential([
    # head
    keras.layers.InputLayer((imageSize, imageSize, channel)),

    # data augmentation
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.2),
    
    # CNN + Transformer
    ConTransformer(imageSize, patchSize=1, dim=128, depth=4, heads=4, mlpDim=512, dropout=0.1),

    # output
    keras.layers.Dense(8, activation='softmax') 
])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0, weight_decay=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
model.summary()

# training
lrScheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
model.fit(
  trainImages,
  epochs=100,
  validation_data=valImages,
  class_weight=weights,
  callbacks=[lrScheduler]
  )
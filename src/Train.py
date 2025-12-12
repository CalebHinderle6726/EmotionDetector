import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from ConvolutionalTransformer import ConTransformer
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# image dimesnions, pixel count and color channels
imageSize = 96
channel = 3     # if channel is 1, add grayscale to dataset loader

# loading train subset of dataset
trainImages = keras.utils.image_dataset_from_directory(
    "data/train",
    validation_split=0.2,
    subset="training",
    seed=554,
    image_size=(imageSize, imageSize),
    batch_size=32,
    )  

# loading validation subset of dataset
valImages = keras.utils.image_dataset_from_directory(
    "data/train",
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


# model
model = keras.Sequential([
    # head
    keras.layers.InputLayer((imageSize, imageSize, channel)),

    # data augmentation
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.05),
    
    # CNN + Transformer
    ConTransformer(imageSize, patchSize=1, dim=128, depth=4, heads=4, mlpDim=256, dropout=0.1),

    # output
    keras.layers.Dense(8, activation='softmax', dtype='float32') 
])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0, weight_decay=1e-4),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
            jit_compile=True)
model.summary()

# training
lrScheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
history = model.fit(
  trainImages,
  epochs=75,
  validation_data=valImages,
  callbacks=[lrScheduler]
  )

# test evaluation
test_loss, test_acc = model.evaluate(testImages, verbose=2)
print(f"Test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}")

# persist training history arrays
historyDict = history.history
np.savez(
    artifactPath / "training_history.npz",
    accuracy=np.array(historyDict.get("accuracy", [])),
    val_accuracy=np.array(historyDict.get("val_accuracy", [])),
    loss=np.array(historyDict.get("loss", [])),
    val_loss=np.array(historyDict.get("val_loss", [])),
    test_accuracy=np.array([test_acc]),
    test_loss=np.array([test_loss]),
)

# plot curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
epochsRange = range(1, len(historyDict.get("accuracy", [])) + 1)

axes[0].plot(epochsRange, historyDict.get("loss", []), label="Train Loss")
axes[0].plot(epochsRange, historyDict.get("val_loss", []), label="Val Loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochsRange, historyDict.get("accuracy", []), label="Train Acc")
axes[1].plot(epochsRange, historyDict.get("val_accuracy", []), label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(artifactPath / "training_curves.png")
plt.close(fig)

# save model
model.save(artifactPath / "model.keras")

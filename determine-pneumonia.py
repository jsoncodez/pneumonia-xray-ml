import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import kagglehub


dataset_id = "paultimothymooney/chest-xray-pneumonia"
download_path = "./data"


os.makedirs("./data", exist_ok=True)


if os.path.exists("./data/chest_xray/test"):
    print("\nDataset looks to be already downloaded, if not delete folders inside data/chest_xray....\n")
else:
    path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia",
        output_dir="./data",
    )
    print(f"Dataset downloaded to {path}")


train_dir = "./data/chest_xray/train"   #train dataset
test_dir = "./data/chest_xray/test"     #test dataset
valid_dir = "./data/chest_xray/val"     #validation dataset


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary',
    shuffle=False
)

        # ------ build model -------

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


        # ----- train model ------
# early stop - prevent overfitting? still seems to be an issue. low test data accuracy
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)


        # ------ eval model on test set---------
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test accuracy: {test_acc:.4f}")


        # ------ plot training history ---------
# Plot train data
plt.figure(figsize=(12, 4))

    #accuracy plotting
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

    #loss plotting
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

        # -------- make prediction future feature to add own image to determine-------------
# from tensorflow.keras.preprocessing import image
#
# # Path to a new image you want to classify
# img_path = 'path_to_new_image.jpg'
#
# # Load and preprocess the image
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = img_array / 255.0  # Normalize the image
#
# # Make a prediction
# prediction = model.predict(img_array)
#
# # Output the result
# if prediction >= 0.5:
#     print("The image shows signs of Pneumonia.")
# else:
#     print("The image is Normal.")

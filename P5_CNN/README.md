### *APPLICATION OF MACHINE LEARNING IN BIOLOGICAL SYSTEMS (ES60011)*
# **Project-5**
#### Convolutional Neural Network


## Usage
1. Open the Jupyter Notebook:
2. The dataset should be present in the same directory as the notebook.
3. `Run All` code blocks
4. Predictions on the testdata is stored in `predictions.txt` as well as printed below the last cell.


> This code provides a complete workflow for training a CNN to classify images, including data preprocessing, model training, evaluation, and prediction.


## 1. Import Library
* Gather all necessary libraries for data manipulation and decision tree modeling

```
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
```

## 2. Define paths
```
dataset_path = 'Dataset2/FNA'
benign_path = os.path.join(dataset_path, 'benign')
malignant_path = os.path.join(dataset_path, 'malignant')
test_path = 'Dataset2/test'
```
* **dataset_path**: Path to the main dataset directory.
* **benign_path**: Path to the directory containing benign images.
* **malignant_path**: Path to the directory containing malignant images.
* **test_path**: Path to the directory containing test images.

## 3. Image parameters
```
img_height, img_width = 150, 150
batch_size = 32
```
* **img_height, img_width**: Dimensions to which all images will be resized.
* **batch_size**: Number of images to be processed in each batch.


### ImageDataGenerator: 
```

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)
```
* **ImageDataGenerator** : Generates batches of tensor image data with real-time data augmentation.
    * **rescale**: Rescales the pixel values from [0, 255] to [0, 1].
    * **validation_split**: Fraction of images reserved for validation.
* flow_from_directory: Takes the path to a directory and generates batches of augmented data.
    * **target_size**: Resizes all images to the specified dimensions.
    * **class_mode**: Specifies the type of label arrays that are returned (binary for binary classification).
    * **subset**: Specifies whether the data is for training or validation.


## 4. Define the CNN model
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

* **Sequential**: Linear stack of layers.
* **Conv2D**: 2D convolution layer.
    * **32, 64, 128**: Number of filters.
    * **(3, 3)**: Kernel size.
    * **activation='relu'**: Activation function.
    * **input_shape**: Shape of the input image.
* **MaxPooling2D**: Max pooling operation for spatial data.
    * **pool_size**: Size of the pooling window.
* **Flatten**: Flattens the input.
* **Dense**: Fully connected layer.
    * **512**: Number of units.
    * **activation='relu'**: Activation function.
* **Dropout**: Dropout layer to prevent overfitting.
    * **0.5**: Dropout rate.
* **Dense**: Output layer.
    * **1**: Number of units.
    * **activation='sigmoid'**: Activation function for binary classification.

## 5. Compile the model
```
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
```
* **compile**: Configures the model for training.
    * **optimizer**: Optimization algorithm (`Adam`).
    * **loss**: Loss function (`binary_crossentropy` for binary classification).
    * **metrics**: List of metrics to be evaluated by the model during training and testing (`accuracy`).


## 6. Train the model
    
```
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10
)
```

* fit: Trains the model for a fixed number of epochs.
    * **train_generator**: Training data.
    * **steps_per_epoch**: Total number of steps (batches of samples) to yield from the generator before declaring one epoch finished.
    * **validation_data**: Validation data.
    * **validation_steps**: Total number of steps (batches of samples) to yield from the generator before stopping validation.
    * **epochs**: Number of epochs to train the model.

## 7. Save the model
```
model.save('model.h5')
```
* **save**: Saves the model to a file.

## 8. Plot training & validation accuracy and loss
```

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

* **history.history**: Dictionary containing the training history.
* **plt.figure**: Creates a new figure.
* **plt.subplot**: Adds a subplot to the current figure.
* **plt.plot**: Plots data.
* **plt.legend**: Adds a legend.
* **plt.title**: Adds a title.
* **plt.show**: Displays the plot.

## 9. Predict on test images
```
test_images = [os.path.join(test_path, img) for img in os.listdir(test_path)]
predictions = []

for img_path in test_images:
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predictions.append('benign' if prediction < 0.5 else 'malignant')
```

* **os.listdir**: Lists the files in a directory.
* **image.load_img**: Loads an image.
* **image.img_to_array**: Converts an image to a numpy array.
* **np.expand_dims**: Expands the shape of an array.
* **model.predict**: Generates output predictions for the input samples

## 10. Save predictions
```
with open('predictions.txt', 'w') as f:
    for img_path, pred in zip(test_images, predictions):
        f.write(f"{os.path.basename(img_path)}: {pred}\n")  
```

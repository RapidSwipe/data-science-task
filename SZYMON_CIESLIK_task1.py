import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop

data_dir = 'data'
class_labels = ['0', 'A', 'B', 'C']

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.1,
    dtype=np.float32
)


train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    classes=class_labels,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    classes=class_labels,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    subset='validation'
)


all_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    classes=class_labels,
    class_mode='categorical',
    batch_size=299 * len(class_labels),
    shuffle=True
)

images, labels = next(all_gen)


from sklearn.model_selection import train_test_split

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)



# Initializing the CNN
classifier = Sequential()

# First convolutional layer
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# First max-pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation='relu'))

# Second max-pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully connected layer
classifier.add(Dense(128, activation='relu'))

# Adding a dropout layer
classifier.add(Dropout(0.5))

# Output layer
classifier.add(Dense(4, activation='softmax'))

# Compiling the CNN
optimizer = RMSprop(learning_rate=0.0001)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


history= classifier.fit_generator(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=25,
    validation_data=val_gen,
    validation_steps=len(val_gen)
    )


#Plot the learning curves for 'loss' and 'accuracy':

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random
import numpy as np


def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_learning_curves(history)


#Test the model on test data and calculate accuracy, precision, recall, and F1-score:

test_pred = np.argmax(classifier.predict(test_images), axis=-1)
test_true = np.argmax(test_labels, axis=-1)

accuracy = accuracy_score(test_true, test_pred)
precision = precision_score(test_true, test_pred, average='weighted')
recall = recall_score(test_true, test_pred, average='weighted')
f1 = f1_score(test_true, test_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=class_labels))


#Display the predictions for 10 random images:

def plot_random_predictions(images, true_labels, pred_labels, class_labels):
    plt.figure(figsize=(15, 7))
    indices = random.sample(range(len(images)), 10)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx])
        plt.xticks([])
        plt.yticks([])
        true_label = class_labels[true_labels[idx]]
        pred_label = class_labels[pred_labels[idx]]
        plt.title(f"True: {true_label}, Pred: {pred_label}")

    plt.show()

plot_random_predictions(test_images, test_true, test_pred, class_labels)



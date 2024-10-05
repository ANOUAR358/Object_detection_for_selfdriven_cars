import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


def createdata(path, path_i):
    images = []
    labels_class = []
    labels_bbox = []
    
    names = os.listdir(path)
    i = os.listdir(path_i)
    files = [os.path.join(path, name) for name in names]

    for text, tn in zip(files, i):
        with open(text) as file:
            image_labels_class = []
            image_labels_bbox = []
            for line in file.readlines():
                vecteur = [0, 0, 0, 0, 0]  
                line = line.split()
                vecteur[0] = int(line[0])  
                vecteur[1:] = [float(x) for x in line[1:]]  
                # Use 0 instead of 5 for padding class labels
                image_labels_class.append(vecteur[0] if vecteur[0] != -1 else 0)
                image_labels_bbox.append(vecteur[1:])  # Add bounding box labels

            labels_class.append(image_labels_class)
            labels_bbox.append(image_labels_bbox)

            image_path = os.path.join(path_i, tn)
            image = cv2.imread(image_path)
            image_r = cv2.resize(image, (255, 255))
            matrix = np.array(image_r)
            images.append(matrix)

    # Pad sequences with a special value (-1)
    padded_labels_class = pad_sequences(labels_class, maxlen=5, padding='post', value=0)
    padded_labels_bbox = pad_sequences(labels_bbox, maxlen=5, padding='post', value=-1)

    return np.array(images), (padded_labels_class, padded_labels_bbox), image_path


def create_detection_model(input_shape, max_objects, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    # Change the output layers to match the correct shape
    class_output = layers.Dense(max_objects * num_classes, activation='softmax')(x)
    bbox_output = layers.Dense(max_objects * 4, activation='sigmoid')(x)
    class_output = layers.Reshape((max_objects, num_classes), name='class_output')(class_output)
    bbox_output = layers.Reshape((max_objects, 4), name='bbox_output')(bbox_output)
    model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    return model


def train_detection_model(model, train_data, val_data, num_epochs, max_objects):
    train_images, (train_labels_class, train_labels_bbox) = train_data
    val_images, (val_labels_class, val_labels_bbox) = val_data

    model.compile(optimizer='adam', 
                  loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': keras.losses.mean_squared_error}, 
                  loss_weights={'class_output': 1.0, 'bbox_output': 1.0},
                  metrics=['accuracy','accuracy'])

    model.fit(train_images, {'class_output': train_labels_class, 'bbox_output': train_labels_bbox}, 
              validation_data=(val_images, {'class_output': val_labels_class, 'bbox_output': val_labels_bbox}),
              epochs=num_epochs)



import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.initializers import VarianceScaling
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Function to start the camera feed
def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def update_frame():
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            root.after(10, update_frame)
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        camera_label.config(image=image)
        camera_label.image = image
        root.after(10, update_frame)

    update_frame()

# Function to load images and labels from a directory
def load_dataset(directory, size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    resized_image = cv2.resize(image, size)
                    images.append(resized_image)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Function to construct the CNN model
def construct_model(n_channels, img_x, img_y, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     input_shape=(img_x, img_y, n_channels),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
    return model

# Function to handle the "Train" button click
def train_model():
    if feature_var.get() == 'HOG' and classification_var.get() == 'CNN':
        print("Training model using HOG and CNN...")

        train_images, train_labels = load_dataset('TrainData/')
        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])
        train_hog_features = extract_hog_features(train_images)

        # Print the shape of train_hog_features to debug
        print(f'Shape of train_hog_features: {train_hog_features.shape}')

        X_train, _, y_train, _ = train_test_split(train_hog_features, int_train_labels, test_size=0.2, random_state=42)

        # Print the shape of X_train before reshaping
        print(f'Shape of X_train before reshaping: {X_train.shape}')

        # Reshape based on the actual feature size
        feature_size = train_hog_features.shape[1]
        X_train = X_train.reshape(-1, 64, 64, 1)  # Assuming grayscale images

        y_train = to_categorical(y_train)

        model = construct_model(1, 64, 64, len(np.unique(train_labels)))
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        loss, accuracy = model.evaluate(X_train, y_train)
        print(f'Train accuracy: {accuracy}')

        # Save the model to the 'TrainData/' directory
        if not os.path.exists('TrainData'):
            os.makedirs('TrainData')
        model.save('TrainData/hog_cnn_model.h5')
        print("Model saved to 'TrainData/hog_cnn_model.h5'")
    else:
        print("Please select HOG for feature extraction and CNN for classification.")

root = tk.Tk()
root.title("Camera Feed with Dropdowns")

label1 = tk.Label(root, text="Select Feature Extraction Method:")
label1.pack()
feature_var = tk.StringVar()
feature_dropdown = ttk.Combobox(root, textvariable=feature_var)
feature_dropdown['values'] = ('HOG', 'Hu', 'Haar')
feature_dropdown.current(0)
feature_dropdown.pack()

label2 = tk.Label(root, text="Select Classification Method:")
label2.pack()
classification_var = tk.StringVar()
classification_dropdown = ttk.Combobox(root, textvariable=classification_var)
classification_dropdown['values'] = ('SVN', 'CNN')
classification_dropdown.current(1)
classification_dropdown.pack()

camera_label = tk.Label(root)
camera_label.pack()

start_camera()

train_button = tk.Button(root, text="Train", command=train_model)
train_button.pack()

test_button = tk.Button(root, text="Test", command=test_model)
test_button.pack()

root.mainloop()
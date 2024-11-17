import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
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

# Function to extract HOG features
def extract_hog_features(images, size=(64, 64)):
    winSize = size
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = []
    for image in images:
        resized_image = cv2.resize(image, size)
        hog_feature = hog.compute(resized_image).flatten()
        hog_features.append(hog_feature)
    return np.array(hog_features)

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

        X_train, X_test, y_train, y_test = train_test_split(train_hog_features, int_train_labels, test_size=0.2, random_state=42)

        # Print the shape of X_train before reshaping
        print(f'Shape of X_train before reshaping: {X_train.shape}')

        # Reshape based on the actual feature size
        feature_size = train_hog_features.shape[1]
        X_train = X_train.reshape(-1, feature_size)
        X_test = X_test.reshape(-1, feature_size)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(feature_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(np.unique(train_labels)), activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Train accuracy: {accuracy}')
    else:
        print("Please select HOG for feature extraction and CNN for classification.")

# Function to handle the "Test" button click
def test_model():
    print("Testing model...")

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
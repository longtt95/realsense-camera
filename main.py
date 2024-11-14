import pyrealsense2 as rs
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.feature import hog
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your pre-trained CNN model
cnn_model = load_model('train/cnn_model.h5')

# Function to start the camera feed
def start_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Start streaming
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def update_frame():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            root.after(10, update_frame)
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Convert the image to a format suitable for Tkinter
        image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Update the image in the label
        camera_label.config(image=image)
        camera_label.image = image

        # Schedule the next frame update
        root.after(10, update_frame)

    update_frame()

# Function to execute HOG feature extraction and CNN classification
def execute():
    if feature_var.get() == 'HOG' and classification_var.get() == 'CNN':
        # Load and preprocess the dataset
        # (Assuming you have a function to load and preprocess your dataset)
        X_train, y_train, X_test, y_test = load_and_preprocess_dataset()

        # Train the CNN model
        history = cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        # Plot accuracy and loss graphs
        plot_accuracy_and_loss(history)

# Function to load and preprocess the dataset
def load_and_preprocess_dataset():
    # Implement your dataset loading and preprocessing here
    # Return X_train, y_train, X_test, y_test
    pass

# Function to plot accuracy and loss graphs
def plot_accuracy_and_loss(history):
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='test accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='test loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to train the model
def train_model():
    # Load and preprocess the dataset
    X_train, y_train, X_test, y_test = load_and_preprocess_dataset()

    # Train the CNN model
    history = cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Plot accuracy and loss graphs
    plot_accuracy_and_loss(history)

# Create the main window
root = tk.Tk()
root.title("Camera Feed with Dropdowns")

# Create the first dropdown menu
label1 = tk.Label(root, text="Select Feature Extraction Method:")
label1.pack()
feature_var = tk.StringVar()
feature_dropdown = ttk.Combobox(root, textvariable=feature_var)
feature_dropdown['values'] = ('HOG', 'Hu', 'Haar')
feature_dropdown.current(0)  # Set the default value
feature_dropdown.pack()

# Create the second dropdown menu
label2 = tk.Label(root, text="Select Classification Method:")
label2.pack()
classification_var = tk.StringVar()
classification_dropdown = ttk.Combobox(root, textvariable=classification_var)
classification_dropdown['values'] = ('SVN', 'CNN')
classification_dropdown.current(0)  # Set the default value
classification_dropdown.pack()

# Create a label to display the camera feed
camera_label = tk.Label(root)
camera_label.pack()

# Create a button to execute some action
execute_button = tk.Button(root, text="Execute", command=execute)
execute_button.pack()

# Create a button to train the model
train_button = tk.Button(root, text="Train", command=train_model)
train_button.pack()

# Start the camera feed automatically
start_camera()

# Run the GUI event loop
root.mainloop()
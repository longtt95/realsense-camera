import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout, concatenate
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
from sklearn import svm
import joblib
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tkinter.font as tkFont
from sklearn.svm import SVC

# Global variables to store frames
color_image_global = None
depth_image_global = None

# Function to start the camera feed
def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def update_frame():
        global color_image_global, depth_image_global
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                root.after(10, update_frame)
                return

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_global = color_image
            depth_image_global = depth_image

            # Create a mask where depth values are within a certain range (e.g., 0 to 1000)
            mask = (depth_image > 0) & (depth_image < 1000)

            # Apply the mask to the depth colormap
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap[~mask] = 0
            images = np.hstack((color_image, depth_colormap))
            image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            camera_label.config(image=image)
            camera_label.image = image
        except RuntimeError as e:
            print(f"Error: {e}")
        finally:
            root.after(10, update_frame)

    update_frame()


# Function to load 2D images and labels from a directory
def load_2d_dataset(directory, size=(64, 64)):
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


# Function to load 3D images and labels from a directory
def load_3d_dataset(directory, size=(64, 64, 64)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = np.load(image_path)  # Assuming 3D images are stored as .npy files
                if image is not None:
                    # Ensure the image has the correct number of dimensions
                    if len(image.shape) == 2:
                        # Resize the 2D image to the target 3D size
                        resized_image = np.zeros(size)
                        for i in range(size[2]):
                            resized_image[:, :, i] = cv2.resize(image, (size[0], size[1]))
                        images.append(resized_image)
                    elif len(image.shape) == 3 and image.shape != size:
                        # Resize the 3D image to the target size
                        resized_image = np.zeros(size)
                        for i in range(min(image.shape[2], size[2])):
                            resized_image[:, :, i] = cv2.resize(image[:, :, i], (size[0], size[1]))
                        images.append(resized_image)
                    else:
                        images.append(image)
                    labels.append(label)
    return np.array(images), np.array(labels)


# Function to check dataset balance
def check_dataset_balance(labels):
    unique, counts = np.unique(labels, return_counts=True)
    balance = dict(zip(unique, counts))
    print("Dataset balance:", balance)
    return balance


# Function to preprocess images
def preprocess_images(images, size=(64, 64)):
    if len(size) == 2:
        preprocessed_images = [cv2.resize(img, size) for img in images]
    elif len(size) == 3:
        preprocessed_images = []
        for img in images:
            resized_image = np.zeros(size)
            for i in range(size[2]):
                resized_image[:, :, i] = cv2.resize(img[:, :, i], (size[0], size[1]))
            preprocessed_images.append(resized_image)
    return np.array(preprocessed_images)


# Function to construct the combined model
def construct_combined_model_hog_cnn(input_shape_2d, input_shape_3d, num_classes):
    # 2D branch for HOG features
    input_2d = Input(shape=(1764,))  # Adjusted to accept flattened HOG features
    x2d = Dense(512, activation='relu')(input_2d)
    x2d = Dropout(0.5)(x2d)

    # 3D branch
    input_3d = Input(shape=input_shape_3d)
    x3d = Conv3D(32, (3, 3, 3), activation='relu', padding='valid', kernel_initializer=VarianceScaling(scale=2.0))(input_3d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x3d = Conv3D(64, (3, 3, 3), activation='relu', padding='valid', kernel_initializer=VarianceScaling(scale=2.0))(x3d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x3d = Flatten()(x3d)

    # Concatenate the outputs of the two branches
    combined = concatenate([x2d, x3d])
    combined = Dense(4096, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(4096, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[input_2d, input_3d], outputs=output)
    sgd = SGD(learning_rate=0.005, decay=1e-6, momentum=0.95, nesterov=True, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def construct_combined_model_hu_cnn(input_shape_2d, input_shape_3d, num_classes):
    # 2D branch for Hu-moments
    input_2d = Input(shape=input_shape_2d)
    x2d = Dense(32, activation='relu')(input_2d)
    x2d = Dropout(0.5)(x2d)

    # 3D branch for Hu-moments
    input_3d = Input(shape=input_shape_3d)
    x3d = Dense(32, activation='relu')(input_3d)
    x3d = Dropout(0.5)(x3d)

    # Concatenate the outputs of the two branches
    combined = concatenate([x2d, x3d])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[input_2d, input_3d], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def construct_combined_model_haar_cnn(input_shape_2d, input_shape_3d, num_classes):
    # 2D branch for Haar features
    input_2d = Input(shape=input_shape_2d)
    x2d = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(scale=2.0), kernel_regularizer=l2(0.01))(input_2d)
    x2d = BatchNormalization()(x2d)
    x2d = MaxPooling2D(pool_size=(2, 2))(x2d)
    x2d = Dropout(0.25)(x2d)
    x2d = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(scale=2.0), kernel_regularizer=l2(0.01))(x2d)
    x2d = BatchNormalization()(x2d)
    x2d = MaxPooling2D(pool_size=(2, 2))(x2d)
    x2d = Dropout(0.25)(x2d)
    x2d = Flatten()(x2d)

    # 3D branch for Haar features
    input_3d = Input(shape=input_shape_3d)
    x3d = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(scale=2.0), kernel_regularizer=l2(0.01))(input_3d)
    x3d = BatchNormalization()(x3d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x3d = Dropout(0.25)(x3d)
    x3d = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=VarianceScaling(scale=2.0), kernel_regularizer=l2(0.01))(x3d)
    x3d = BatchNormalization()(x3d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x3d = Dropout(0.25)(x3d)
    x3d = Flatten()(x3d)

    # Concatenate the outputs of the two branches
    combined = concatenate([x2d, x3d])
    combined = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[input_2d, input_3d], outputs=output)
    sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def construct_cnn_model(input_shape_2d, input_shape_3d, num_classes):
    # 2D branch
    input_2d = Input(shape=input_shape_2d)
    x2d = Conv2D(32, (3, 3), activation='relu', padding='same')(input_2d)
    x2d = MaxPooling2D(pool_size=(2, 2))(x2d)
    x2d = Conv2D(64, (3, 3), activation='relu', padding='same')(x2d)
    x2d = MaxPooling2D(pool_size=(2, 2))(x2d)
    x2d = Flatten()(x2d)

    # 3D branch
    input_3d = Input(shape=input_shape_3d)
    x3d = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_3d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x3d = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x3d)
    x3d = MaxPooling3D(pool_size=(2, 2, 2))(x3d)
    x3d = Flatten()(x3d)

    # Concatenate the outputs of the two branches
    combined = concatenate([x2d, x3d])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[input_2d, input_3d], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def extract_hu_moments(image):
    """
    Extract Hu moments from an image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Image is already in grayscale
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# Function to handle the "Train" button click
def train_model():
    status_label.config(text="Đang training dữ liệu. Vui lòng đợi...")
    root.update_idletasks()  # Update the GUI to reflect the change immediately

    print("Training model...")
    feature = feature_var.get()
    classification = classification_var.get()

    if classification == 'CNN':
        train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
        train_images_3d, train_labels_3d = load_3d_dataset('TrainData/3D/')

        # Check dataset balance
        balance = check_dataset_balance(train_labels)

        # Preprocess images
        train_images_2d = preprocess_images(train_images_2d)
        train_images_3d = preprocess_images(train_images_3d, size=(64, 64, 64))

        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])
        X_train_2d, X_val_2d, y_train, y_val = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)
        X_train_3d, X_val_3d, _, _ = train_test_split(train_images_3d, int_train_labels, test_size=0.2, random_state=42)
        y_train = to_categorical(y_train, num_classes=len(label_to_int))
        y_val = to_categorical(y_val, num_classes=len(label_to_int))

        # Reshape the 3D images to match the expected input structure
        X_train_3d = X_train_3d.reshape(-1, 64, 64, 64, 1)
        X_val_3d = X_val_3d.reshape(-1, 64, 64, 64, 1)

        # Construct and train the model
        model = construct_cnn_model((64, 64, 1), (64, 64, 64, 1), len(label_to_int))
        model.fit([X_train_2d, X_train_3d], y_train, validation_data=([X_val_2d, X_val_3d], y_val), epochs=5, batch_size=64)
        model.save('TrainData/cnn_model.keras')
        print("Model saved to 'TrainData/cnn_model.keras'")
    # elif feature == 'HOG' and classification == 'CNN':
    #     train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
    #     train_images_3d, _ = load_3d_dataset('TrainData/3D/')
    #
    #     # Check dataset balance
    #     balance = check_dataset_balance(train_labels)
    #
    #     # Preprocess images
    #     train_images_2d = preprocess_images(train_images_2d)
    #     train_images_3d = preprocess_images(train_images_3d, size=(64, 64, 64))
    #
    #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    #     int_train_labels = np.array([label_to_int[label] for label in train_labels])
    #     X_train_2d, X_val_2d, y_train, y_val = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)
    #     X_train_3d, X_val_3d, _, _ = train_test_split(train_images_3d, int_train_labels, test_size=0.2, random_state=42)
    #     y_train = to_categorical(y_train, num_classes=len(label_to_int))
    #     y_val = to_categorical(y_val, num_classes=len(label_to_int))
    #
    #     # Extract HOG features for 2D images
    #     hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    #     X_train_hog_2d = np.array([hog.compute(cv2.resize(img, (64, 64))).flatten() for img in X_train_2d])
    #     X_val_hog_2d = np.array([hog.compute(cv2.resize(img, (64, 64))).flatten() for img in X_val_2d])
    #
    #     # Normalize HOG features
    #     X_train_hog_2d = X_train_hog_2d / np.linalg.norm(X_train_hog_2d, axis=1, keepdims=True)
    #     X_val_hog_2d = X_val_hog_2d / np.linalg.norm(X_val_hog_2d, axis=1, keepdims=True)
    #
    #     # Check for NaN values in the dataset
    #     if np.isnan(X_train_hog_2d).any() or np.isnan(X_val_hog_2d).any():
    #         print("NaN values found in the dataset. Please check the data preprocessing steps.")
    #         return
    #
    #     # Reshape the 3D images to match the expected input structure
    #     X_train_3d = X_train_3d.reshape(-1, 64, 64, 64, 1)
    #     X_val_3d = X_val_3d.reshape(-1, 64, 64, 64, 1)
    #
    #     # Construct and train the model
    #     model = construct_combined_model_hog_cnn((1764,), (64, 64, 64, 1), len(label_to_int))
    #     model.fit([X_train_hog_2d, X_train_3d], y_train, validation_data=([X_val_hog_2d, X_val_3d], y_val), epochs=16, batch_size=64)
    #     model.save('TrainData/hog_cnn_model.keras')
    #     print("Model saved to 'TrainData/hog_cnn_model.keras'")

    elif feature == 'HOG' and classification == 'SVM':
        # Load 2D dataset
        train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')

        # Check dataset balance
        balance = check_dataset_balance(train_labels)

        # Extract HOG features for 2D images
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        train_images_2d_hog = np.array([hog.compute(cv2.resize(img, (64, 64))).flatten() for img in train_images_2d])

        # Convert labels to integers
        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])

        # Train SVM model
        svm_model = svm.SVC(kernel='linear', probability=True)
        svm_model.fit(train_images_2d_hog, int_train_labels)
        joblib.dump(svm_model, 'TrainData/hog_svm_model.pkl')
        print("Model saved to 'TrainData/hog_svm_model.pkl'")

    # elif feature == 'Hu' and classification == 'CNN':
    #     train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
    #     train_images_3d, _ = load_3d_dataset('TrainData/3D/')
    #
    #     # Ensure the number of 2D and 3D images are the same
    #     min_samples = min(len(train_images_2d), len(train_images_3d))
    #     train_images_2d = train_images_2d[:min_samples]
    #     train_images_3d = train_images_3d[:min_samples]
    #     train_labels = train_labels[:min_samples]
    #
    #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    #     int_train_labels = np.array([label_to_int[label] for label in train_labels])
    #
    #     X_train_2d, X_val_2d, y_train, y_val = train_test_split(train_images_2d, int_train_labels, test_size=0.2,
    #                                                             random_state=42)
    #     X_train_3d, X_val_3d, _, _ = train_test_split(train_images_3d, int_train_labels, test_size=0.2, random_state=42)
    #
    #     X_train_hu_2d = np.array([cv2.HuMoments(cv2.moments(img)).flatten() for img in X_train_2d])
    #     X_val_hu_2d = np.array([cv2.HuMoments(cv2.moments(img)).flatten() for img in X_val_2d])
    #     X_train_hu_3d = np.array([cv2.HuMoments(cv2.moments(img[:, :, 0])).flatten() for img in X_train_3d])
    #     X_val_hu_3d = np.array([cv2.HuMoments(cv2.moments(img[:, :, 0])).flatten() for img in X_val_3d])
    #
    #     y_train = to_categorical(y_train, num_classes=len(label_to_int))
    #     y_val = to_categorical(y_val, num_classes=len(label_to_int))
    #
    #     model = construct_combined_model_hu_cnn((7,), (7,), len(np.unique(train_labels)))
    #     model.fit([X_train_hu_2d, X_train_hu_3d], y_train, validation_data=([X_val_hu_2d, X_val_hu_3d], y_val),
    #               epochs=10, batch_size=64)
    #     model.save('TrainData/hu_cnn_model.h5')
    #     print("Model saved to 'TrainData/hu_cnn_model.h5'")

    elif feature == 'Hu' and classification == 'SVM':
        # Load 2D dataset
        train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')

        # Check dataset balance
        balance = check_dataset_balance(train_labels)

        # Extract Hu moments for 2D images
        train_images_2d_hu = np.array([extract_hu_moments(img) for img in train_images_2d])

        # Convert labels to integers
        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])

        # Train SVM model
        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(train_images_2d_hu, int_train_labels)
        joblib.dump(svm_model, 'TrainData/hu_svm_model.pkl')
        print("Model saved to 'TrainData/hu_svm_model.pkl'")

    # elif feature == 'Haar' and classification == 'CNN':
    #     train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
    #     train_images_3d, _ = load_3d_dataset('TrainData/3D/')
    #
    #     min_samples = min(len(train_images_2d), len(train_images_3d))
    #     train_images_2d = train_images_2d[:min_samples]
    #     train_images_3d = train_images_3d[:min_samples]
    #     train_labels = train_labels[:min_samples]
    #
    #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    #     int_train_labels = np.array([label_to_int[label] for label in train_labels])
    #
    #     X_train_2d, X_val_2d, y_train, y_val = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)
    #     X_train_3d, X_val_3d, _, _ = train_test_split(train_images_3d, int_train_labels, test_size=0.2, random_state=42)
    #
    #     X_train_2d = X_train_2d.reshape(-1, 64, 64, 1)
    #     X_val_2d = X_val_2d.reshape(-1, 64, 64, 1)
    #
    #     datagen = ImageDataGenerator(
    #         rotation_range=20,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         horizontal_flip=True
    #     )
    #     datagen.fit(X_train_2d)
    #
    #     X_train_haar_2d = np.array([cv2.integral(img) for img in X_train_2d])
    #     X_val_haar_2d = np.array([cv2.integral(img) for img in X_val_2d])
    #
    #     epsilon = 1e-10
    #     X_train_haar_2d = X_train_haar_2d / (np.linalg.norm(X_train_haar_2d, axis=1, keepdims=True) + epsilon)
    #     X_val_haar_2d = X_val_haar_2d / (np.linalg.norm(X_val_haar_2d, axis=1, keepdims=True) + epsilon)
    #
    #     if np.isnan(X_train_haar_2d).any() or np.isnan(X_val_haar_2d).any():
    #         print("NaN values found in the dataset. Please check the data preprocessing steps.")
    #         return
    #
    #     y_train = to_categorical(y_train, num_classes=len(label_to_int))
    #     y_val = to_categorical(y_val, num_classes=len(label_to_int))
    #
    #     X_train_haar_2d = X_train_haar_2d.reshape(-1, 65, 65, 1)
    #     X_val_haar_2d = X_val_haar_2d.reshape(-1, 65, 65, 1)
    #
    #     X_train_3d = X_train_3d.reshape(-1, 64, 64, 64, 1)
    #     X_val_3d = X_val_3d.reshape(-1, 64, 64, 64, 1)
    #
    #     model = construct_combined_model_haar_cnn((65, 65, 1), (64, 64, 64, 1), len(label_to_int))
    #
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #
    #     # Create TensorFlow datasets
    #     train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_haar_2d, X_train_3d), y_train))
    #     val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_haar_2d, X_val_3d), y_val))
    #
    #     # Shuffle and batch the datasets
    #     train_dataset = train_dataset.shuffle(buffer_size=len(X_train_haar_2d)).batch(64)
    #     val_dataset = val_dataset.batch(64)
    #
    #     model.fit(train_dataset,
    #               validation_data=val_dataset,
    #               epochs=10, callbacks=[early_stopping])
    #     model.save('TrainData/haar_cnn_model.keras')
    #     print("Model saved to 'TrainData/haar_cnn_model.keras'")

    elif feature == 'Haar' and classification == 'SVM':
        train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])
        X_train_2d, _, y_train, _ = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)

        # Extract Haar features for 2D images and flatten them
        X_train_haar_2d = [cv2.integral(img).flatten() for img in X_train_2d]

        svm_model = svm.SVC(kernel='linear', probability=True)
        svm_model.fit(X_train_haar_2d, y_train)
        joblib.dump(svm_model, 'TrainData/haar_svm_model.pkl')
        print("Model saved to 'TrainData/haar_svm_model.pkl'")

    status_label.config(text="Đã hoàn thành training dữ liệu.")
    # elif feature_var.get() == 'HOG' and classification_var.get() == 'SVM':
    #     # Load datasets
    #     train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
    #     train_images_3d, _ = load_3d_dataset('TrainData/3D/')
    #
    #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    #     int_train_labels = np.array([label_to_int[label] for label in train_labels])
    #
    #     X_train_2d, _, y_train, _ = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)
    #     X_train_3d, _, _, _ = train_test_split(train_images_3d, int_train_labels, test_size=0.2, random_state=42)
    #
    #     # Resize images to 64x64 before extracting HOG features
    #     X_train_2d_resized = [cv2.resize(img, (64, 64)) for img in X_train_2d]
    #
    #     # Ensure images are single-channel (grayscale)
    #     X_train_2d_resized = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in X_train_2d_resized]
    #
    #     # Extract HOG features for 2D images
    #     hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    #     X_train_2d_hog = np.array([hog.compute(img).flatten() for img in X_train_2d_resized])
    #
    #     # Combine 2D HOG features and 3D images
    #     X_train_combined = np.hstack((X_train_2d_hog, X_train_3d.reshape(X_train_3d.shape[0], -1)))
    #
    #     # Train SVM model
    #     svm_model = svm.SVC(kernel='linear', probability=True)
    #     svm_model.fit(X_train_combined, y_train)
    #
    #     # Save the SVM model
    #     if not os.path.exists('TrainData'):
    #         os.makedirs('TrainData')
    #     joblib.dump(svm_model, 'TrainData/hog_svm_model.pkl')
    #     print("Model saved to 'TrainData/hog_svm_model.pkl'")

def test_model():
    print("Testing model...")
    status_label.config(text="Đang dự đoán. Vui lòng đợi...")
    root.update_idletasks()  # Update the GUI to reflect the change immediately
    feature = feature_var.get()
    classification = classification_var.get()

    file_path = filedialog.askopenfilename(title="Select an image",
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        print("No file selected")
        return

    color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if color_image is None:
        print("Failed to load image")
        return

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))

    if classification == 'CNN':
        model_path = 'TrainData/cnn_model.keras'
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return
        model = load_model(model_path)
        resized_image = resized_image.reshape(1, 64, 64, 1)
        dummy_3d_input = np.zeros((1, 64, 64, 64, 1))
        prediction = model.predict([resized_image, dummy_3d_input])
    elif feature == 'HOG' and classification == 'SVM':
        model_path = 'TrainData/hog_svm_model.pkl'
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        hog_feature = hog.compute(resized_image).flatten()

        svm_model = joblib.load(model_path)
        prediction = svm_model.predict_proba([hog_feature])
    elif feature == 'Hu' and classification == 'SVM':
        model_path = 'TrainData/hu_svm_model.pkl'
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return
        hu_feature = extract_hu_moments(color_image)
        svm_model = joblib.load(model_path)
        prediction = svm_model.predict_proba([hu_feature])
    elif feature == 'Haar' and classification == 'SVM':
        model_path = 'TrainData/haar_svm_model.pkl'
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return
        haar_feature = cv2.integral(resized_image).flatten()
        svm_model = joblib.load(model_path)
        prediction = svm_model.predict_proba([haar_feature])
    else:
        print(f"Feature extraction method {feature} and classification method {classification} not implemented")
        return

    predicted_label = np.argmax(prediction, axis=1)
    accuracy = np.max(prediction) * 100
    train_images, train_labels = load_2d_dataset('TrainData/2D/')
    label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    predicted_label_name = int_to_label[predicted_label[0]]
    print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
    label_result.config(text=f'Dự đoán: {predicted_label_name}')

    # Resize the image to 100px width while maintaining aspect ratio
    height, width = color_image.shape[:2]
    new_width = 100
    new_height = int((new_width / width) * height)
    resized_color_image = cv2.resize(color_image, (new_width, new_height))

    # Display the image in the GUI
    image = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    uploaded_image_label.config(image=image)
    uploaded_image_label.image = image
    status_label.config(text="")

# def test_model():
#     print("Testing model...")
#     status_label.config(text="Đang dự đoán. Vui lòng đợi...")
#     root.update_idletasks()  # Update the GUI to reflect the change immediately
#     feature = feature_var.get()
#     classification = classification_var.get()
#
#     if classification == 'CNN':
#         file_path = filedialog.askopenfilename(title="Select an image",
#                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#         if not file_path:
#             print("No file selected")
#             return
#         color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#         if color_image is None:
#             print("Failed to load image")
#             return
#         gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#         resized_image = cv2.resize(gray_image, (64, 64))
#         model = load_model('TrainData/cnn_model.keras')
#         resized_image = resized_image.reshape(1, 64, 64, 1)
#         dummy_3d_input = np.zeros((1, 64, 64, 64, 1))
#         prediction = model.predict([resized_image, dummy_3d_input])
#         predicted_label = np.argmax(prediction, axis=1)
#         accuracy = np.max(prediction) * 100
#         train_images, train_labels = load_2d_dataset('TrainData/2D/')
#         label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#         int_to_label = {idx: label for label, idx in label_to_int.items()}
#         predicted_label_name = int_to_label[predicted_label[0]]
#         print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#         label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     # elif feature == 'HOG' and classification == 'CNN':
#     #     file_path = filedialog.askopenfilename(title="Select an image",
#     #                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#     #     if not file_path:
#     #         print("No file selected")
#     #         return
#     #     color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#     #     if color_image is None:
#     #         print("Failed to load image")
#     #         return
#     #     gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#     #     resized_image = cv2.resize(gray_image, (64, 64))
#     #     hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
#     #     hog_feature = hog.compute(resized_image).flatten()
#     #     model = load_model('TrainData/hog_cnn_model.keras')
#     #     hog_feature = hog_feature.reshape(1, 1764)
#     #     dummy_3d_input = np.zeros((1, 64, 64, 64, 1))
#     #     prediction = model.predict([hog_feature, dummy_3d_input])
#     #     predicted_label = np.argmax(prediction, axis=1)
#     #     accuracy = np.max(prediction) * 100
#     #     train_images, train_labels = load_2d_dataset('TrainData/2D/')
#     #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#     #     int_to_label = {idx: label for label, idx in label_to_int.items()}
#     #     predicted_label_name = int_to_label[predicted_label[0]]
#     #     print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#     #     label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     elif feature == 'HOG' and classification == 'SVM':
#         file_path = filedialog.askopenfilename(title="Select an image",
#                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#         if not file_path:
#             print("No file selected")
#             return
#         color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#         if color_image is None:
#             print("Failed to load image")
#             return
#         gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#         resized_image = cv2.resize(gray_image, (64, 64))
#         hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
#         hog_feature = hog.compute(resized_image).flatten()
#         svm_model = joblib.load('TrainData/hog_svm_model.pkl')
#         prediction = svm_model.predict_proba([hog_feature])
#         predicted_label = np.argmax(prediction, axis=1)
#         accuracy = np.max(prediction) * 100
#         train_images, train_labels = load_2d_dataset('TrainData/2D/')
#         label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#         int_to_label = {idx: label for label, idx in label_to_int.items()}
#         predicted_label_name = int_to_label[predicted_label[0]]
#         print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#         label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     # elif feature == 'Hu' and classification == 'CNN':
#     #     file_path = filedialog.askopenfilename(title="Select an image",
#     #                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#     #     if not file_path:
#     #         print("No file selected")
#     #         return
#     #     color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#     #     if color_image is None:
#     #         print("Failed to load image")
#     #         return
#     #     gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#     #     resized_image = cv2.resize(gray_image, (64, 64))
#     #     hu_moments = cv2.HuMoments(cv2.moments(resized_image)).flatten()
#     #     model = load_model('TrainData/hu_cnn_model.h5')
#     #     hu_moments = hu_moments.reshape(1, 7)
#     #     prediction = model.predict([hu_moments, hu_moments])
#     #     predicted_label = np.argmax(prediction, axis=1)
#     #     accuracy = np.max(prediction) * 100
#     #     train_images, train_labels = load_2d_dataset('TrainData/2D/')
#     #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#     #     int_to_label = {idx: label for label, idx in label_to_int.items()}
#     #     predicted_label_name = int_to_label[predicted_label[0]]
#     #     print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#     #     label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     elif feature == 'Hu' and classification == 'SVM':
#         file_path = filedialog.askopenfilename(title="Select an image",
#                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#         if not file_path:
#             print("No file selected")
#             return
#         color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#         if color_image is None:
#             print("Failed to load image")
#             return
#         gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#         resized_image = cv2.resize(gray_image, (64, 64))
#         hu_moments = cv2.HuMoments(cv2.moments(resized_image)).flatten()
#         svm_model = joblib.load('TrainData/hu_svm_model.pkl')
#         prediction = svm_model.predict_proba([hu_moments])
#         predicted_label = np.argmax(prediction, axis=1)
#         accuracy = np.max(prediction) * 100
#         train_images, train_labels = load_2d_dataset('TrainData/2D/')
#         label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#         int_to_label = {idx: label for label, idx in label_to_int.items()}
#         predicted_label_name = int_to_label[predicted_label[0]]
#         print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#         label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     # elif feature == 'Haar' and classification == 'CNN':
#     #     file_path = filedialog.askopenfilename(title="Select an image",
#     #                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#     #     if not file_path:
#     #         print("No file selected")
#     #         return
#     #     color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#     #     if color_image is None:
#     #         print("Failed to load image")
#     #         return
#     #     gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#     #     resized_image = cv2.resize(gray_image, (64, 64))
#     #     haar_feature = cv2.integral(resized_image)
#     #     haar_feature = haar_feature.reshape(1, 65, 65, 1)  # Correctly reshape Haar features
#     #     dummy_3d_input = np.zeros((1, 64, 64, 64, 1))
#     #     model = load_model('TrainData/haar_cnn_model.keras')
#     #     prediction = model.predict([haar_feature, dummy_3d_input])
#     #     predicted_label = np.argmax(prediction, axis=1)
#     #     accuracy = np.max(prediction) * 100
#     #     train_images, train_labels = load_2d_dataset('TrainData/2D/')
#     #     label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#     #     int_to_label = {idx: label for label, idx in label_to_int.items()}
#     #     predicted_label_name = int_to_label[predicted_label[0]]
#     #     print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#     #     label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     elif feature == 'Haar' and classification == 'SVM':
#         file_path = filedialog.askopenfilename(title="Select an image",
#                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
#         if not file_path:
#             print("No file selected")
#             return
#         color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
#         if color_image is None:
#             print("Failed to load image")
#             return
#         gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#         resized_image = cv2.resize(gray_image, (64, 64))
#         haar_feature = cv2.integral(resized_image).flatten()  # Flatten the Haar features
#         svm_model = joblib.load('TrainData/haar_svm_model.pkl')
#         prediction = svm_model.predict_proba([haar_feature])
#         predicted_label = np.argmax(prediction, axis=1)
#         accuracy = np.max(prediction) * 100
#         train_images, train_labels = load_2d_dataset('TrainData/2D/')
#         label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
#         int_to_label = {idx: label for label, idx in label_to_int.items()}
#         predicted_label_name = int_to_label[predicted_label[0]]
#         print(f'Predicted label: {predicted_label_name}, Accuracy: {accuracy:.2f}%')
#         label_result.config(text=f'Dự đoán: {predicted_label_name}')
#
#     # Resize the image to 100px width while maintaining aspect ratio
#     height, width = color_image.shape[:2]
#     new_width = 100
#     new_height = int((new_width / width) * height)
#     resized_color_image = cv2.resize(color_image, (new_width, new_height))
#
#     # Display the image in the GUI
#     image = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image)
#     image = ImageTk.PhotoImage(image)
#     uploaded_image_label.config(image=image)
#     uploaded_image_label.image = image
#     status_label.config(text="")

# Function to load a 3D image from the TrainData/3D/ directory
def load_3d_image(file_path, size=(64, 64, 64)):
    image = np.load(file_path)
    if image.shape != size:
        resized_image = np.zeros(size)
        for i in range(min(image.shape[2], size[2])):
            resized_image[:, :, i] = cv2.resize(image[:, :, i], (size[0], size[1]))
        return resized_image
    return image


# Function to combine HOG features and 3D image
def combine_hog_and_3d(hog_feature, depth_image_path):
    # Load and resize the 3D image
    depth_image = load_3d_image(depth_image_path)
    depth_image_resized = depth_image.flatten()

    # Combine HOG features and 3D image
    combined_features = np.hstack((hog_feature, depth_image_resized))
    return combined_features


# Function to convert 2D images to 3D
# def convert_to_3d():
#     input_dir = 'TrainData/2D/'
#     output_dir = 'TrainData/3D_G/'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for label in os.listdir(input_dir):
#         label_path = os.path.join(input_dir, label)
#         if os.path.isdir(label_path):
#             output_label_path = os.path.join(output_dir, label)
#             if not os.path.exists(output_label_path):
#                 os.makedirs(output_label_path)
#             for image_name in os.listdir(label_path):
#                 image_path = os.path.join(label_path, image_name)
#                 color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                 if color_image is not None:
#                     # Initialize mask for GrabCut
#                     mask = np.zeros(color_image.shape[:2], np.uint8)
#                     bgd_model = np.zeros((1, 65), np.float64)
#                     fgd_model = np.zeros((1, 65), np.float64)
#
#                     # Define a rectangle around the object
#                     rect = (10, 10, color_image.shape[1] - 10, color_image.shape[0] - 10)
#
#                     # Apply GrabCut algorithm
#                     cv2.grabCut(color_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
#                     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#                     color_image = color_image * mask2[:, :, np.newaxis]
#
#                     # Save the 3D image
#                     output_image_path = os.path.join(output_label_path, os.path.splitext(image_name)[0] + '.jpg')
#                     cv2.imwrite(output_image_path, color_image)
#
#     print("Conversion to 3D completed.")

def convert_to_3d():
    input_dir = 'TrainData/2D/'
    output_dir = 'TrainData/3D/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if os.path.isdir(label_path):
            output_label_path = os.path.join(output_dir, label)
            if not os.path.exists(output_label_path):
                os.makedirs(output_label_path)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if color_image is not None:
                    # Initialize mask for GrabCut
                    mask = np.zeros(color_image.shape[:2], np.uint8)
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)

                    # Define a rectangle around the object
                    rect = (10, 10, color_image.shape[1] - 10, color_image.shape[0] - 10)

                    # Apply GrabCut algorithm
                    cv2.grabCut(color_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    color_image = color_image * mask2[:, :, np.newaxis]

                    # Save the 3D image as .npy file
                    output_image_path = os.path.join(output_label_path, os.path.splitext(image_name)[0] + '.npy')
                    np.save(output_image_path, color_image)

    print("Conversion to 3D completed.")


def get_next_filename(directory, extension):
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if not files:
        return 1
    numbers = [int(os.path.splitext(f)[0]) for f in files]
    return max(numbers) + 1


def capture_image():
    global color_image_global, depth_image_global
    label = label_input.get()
    if not label:
        print("Please enter a label")
        return

    # Create directories if they don't exist
    dir_2d = os.path.join('TrainData/2D', label)
    dir_3d = os.path.join('TrainData/3D', label)
    os.makedirs(dir_2d, exist_ok=True)
    os.makedirs(dir_3d, exist_ok=True)

    if color_image_global is None or depth_image_global is None:
        print("No frames available to capture")
        return

    # Get the next available filename number
    next_number_2d = get_next_filename(dir_2d, '.jpg')
    next_number_3d = get_next_filename(dir_3d, '.npy')

    # Save 2D image
    image_2d_path = os.path.join(dir_2d, f'{next_number_2d}.jpg')
    cv2.imwrite(image_2d_path, color_image_global)

    # Save 3D image
    image_3d_path = os.path.join(dir_3d, f'{next_number_3d}.npy')
    np.save(image_3d_path, depth_image_global)

    print(f"Images saved to {dir_2d} and {dir_3d}")


root = tk.Tk()
root.title("Xây dựng mô hình học máy")

# Set font size
font = tkFont.Font(size=12)
font_bold = tkFont.Font(size=12, weight="bold")

label1 = tk.Label(root, text="XÂY DỰNG MÔ HÌNH HỌC MÁY", font=font_bold)
label1.pack()
# Create a frame for the buttons and dropdowns
frame = tk.Frame(root)
frame.pack()

def add_placeholder(entry, placeholder):
    def on_focus_in(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg='black')

    def on_focus_out(event):
        if entry.get() == '':
            entry.insert(0, placeholder)
            entry.config(fg='grey')

    entry.insert(0, placeholder)
    entry.config(fg='grey')
    entry.bind('<FocusIn>', on_focus_in)
    entry.bind('<FocusOut>', on_focus_out)

# Add an input field for the label
label_input = tk.Entry(frame, font=font)
label_input.pack(side=tk.LEFT, padx=5, pady=5)
add_placeholder(label_input, "Nhập label")

# Add a capture button to the frame
capture_button = tk.Button(frame, text="Chụp ảnh", command=capture_image, font=font)
capture_button.pack(side=tk.LEFT, padx=5, pady=5)

# Add a label and dropdown for feature extraction method
label1 = tk.Label(frame, text="Chọn trích xuất đặc trưng:", font=font)
label1.pack(side=tk.LEFT, padx=5, pady=5)
feature_var = tk.StringVar()
feature_dropdown = ttk.Combobox(frame, textvariable=feature_var, font=font)
feature_dropdown['values'] = ('HOG', 'Hu', 'Haar')
feature_dropdown.current(0)
feature_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

# Add a label and dropdown for classification method
label2 = tk.Label(frame, text="Chọn phân lớp:", font=font)
label2.pack(side=tk.LEFT, padx=5, pady=5)
classification_var = tk.StringVar()
classification_dropdown = ttk.Combobox(frame, textvariable=classification_var, font=font)
classification_dropdown['values'] = ('SVM', 'CNN', 'NN')
classification_dropdown.current(1)
classification_dropdown.pack(side=tk.LEFT, padx=5, pady=5)


# # Add the "Convert 3D" button to the frame
# convert_button = tk.Button(frame, text="Convert 3D", command=convert_to_3d, font=font)
# convert_button.pack(side=tk.LEFT, padx=5, pady=5)

# Add a train button to the frame
train_button = tk.Button(frame, text="Train", command=train_model, font=font)
train_button.pack(side=tk.LEFT, padx=5, pady=5)

# Add a test button to the frame
test_button = tk.Button(frame, text="Test", command=test_model, font=font)
test_button.pack(side=tk.LEFT, padx=5, pady=5)

# Add the status label below the frame
status_font = tkFont.Font(size=14, weight="bold", slant="italic")
status_label = tk.Label(root, text="", font=status_font, fg="red")
status_label.pack()

# Add a label to display the prediction result
label_result = tk.Label(root, text="", font=font)
label_result.pack()

camera_label = tk.Label(root)
camera_label.pack()

start_camera()

# Add a label to display the uploaded image
uploaded_image_label = tk.Label(root)
uploaded_image_label.pack()

# Add a label to display the prediction result
label_result = tk.Label(root, text="", font=font)
label_result.pack()

root.mainloop()
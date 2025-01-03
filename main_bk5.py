import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout, concatenate
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.initializers import VarianceScaling
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Function to start the camera feed
def start_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def update_frame():
        try:
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
                    resized_image = cv2.resize(image, (size[0], size[1]))  # Resize each slice
                    images.append(resized_image)
                    labels.append(label)
    return np.array(images), np.array(labels)


# Function to construct the combined model
# Function to construct the combined model with additional regularization
def construct_combined_model(input_shape_2d, input_shape_3d, num_classes):
    # 2D branch
    input_2d = Input(shape=input_shape_2d)
    x2d = Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_initializer=VarianceScaling(scale=2.0))(input_2d)
    x2d = MaxPooling2D(pool_size=(2, 2))(x2d)
    x2d = Conv2D(64, (3, 3), activation='relu', padding='valid', kernel_initializer=VarianceScaling(scale=2.0))(x2d)
    x2d = MaxPooling2D(pool_size=(2, 2))(x2d)
    x2d = Flatten()(x2d)

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
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# Function to handle the "Train" button click
def train_model():
    print("Training model using 2D and 3D images...")

    if feature_var.get() == 'HOG' and classification_var.get() == 'CNN':
        # Load datasets
        train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')
        train_images_3d, _ = load_3d_dataset('TrainData/3D/')

        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])

        X_train_2d, X_val_2d, y_train, y_val = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)
        X_train_3d, X_val_3d, _, _ = train_test_split(train_images_3d, int_train_labels, test_size=0.2, random_state=42)

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        model = construct_combined_model((64, 64, 1), (64, 64, 64, 1), len(np.unique(train_labels)))

        # Add early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        model.fit([X_train_2d, X_train_3d], y_train, epochs=100, batch_size=32, validation_data=([X_val_2d, X_val_3d], y_val), callbacks=[early_stopping, reduce_lr])
        loss, accuracy = model.evaluate([X_val_2d, X_val_3d], y_val)
        print(f'Validation accuracy HOG-CNN: {accuracy}')

        # Save the model to the 'TrainData/' directory
        if not os.path.exists('TrainData'):
            os.makedirs('TrainData')
        model.save('TrainData/hog_cnn_model.keras')
        print("Model saved to 'TrainData/hog_cnn_model.keras'")

    elif feature_var.get() == 'HOG' and classification_var.get() == 'SVM':
        # Load datasets
        train_images_2d, train_labels = load_2d_dataset('TrainData/2D/')

        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_train_labels = np.array([label_to_int[label] for label in train_labels])

        X_train_2d, X_val_2d, y_train, y_val = train_test_split(train_images_2d, int_train_labels, test_size=0.2, random_state=42)

        # Extract HOG features
        hog = cv2.HOGDescriptor()
        X_train_hog = np.array([hog.compute(img).flatten() for img in X_train_2d])
        X_val_hog = np.array([hog.compute(img).flatten() for img in X_val_2d])

        # Train SVM model with hyperparameter tuning
        svm = SVC(kernel='linear', probability=True, C=1.0)
        svm.fit(X_train_hog, y_train)

        # Evaluate the model
        y_pred = svm.predict(X_val_hog)
        accuracy = accuracy_score(y_val, y_pred)
        print(f'Validation accuracy HOG-SVM: {accuracy}')

        # Save the SVM model
        joblib.dump(svm, 'TrainData/hog_svm_model.h5')
        print("Model saved to 'TrainData/hog_svm_model.h5'")


# Function to handle the "Test" button click
def test_model():
    print("Testing model...")
    if feature_var.get() == 'HOG' and classification_var.get() == 'CNN':
        # Capture a single frame from the camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        pipeline.stop()

        if not color_frame:
            print("Failed to capture image")
            return

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (64, 64))

        # Extract HOG features
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog_feature = hog.compute(resized_image).flatten()

        # Load the trained model
        model = load_model('TrainData/hog_cnn_model.h5')

        # Reshape and predict
        hog_feature = hog_feature.reshape(1, -1)
        prediction = model.predict(hog_feature)
        predicted_label = np.argmax(prediction, axis=1)

        # Map the predicted label back to the original label
        train_images, train_labels = load_2d_dataset('TrainData2D/')
        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_to_label = {idx: label for label, idx in label_to_int.items()}
        predicted_label_name = int_to_label[predicted_label[0]]

        print(f'Predicted label: {predicted_label_name}')

    elif feature_var.get() == 'HOG' and classification_var.get() == 'SVM':
        # Capture a single frame from the camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        pipeline.stop()

        if not color_frame:
            print("Failed to capture image")
            return

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (64, 64))

        # Extract HOG features
        hog = cv2.HOGDescriptor()
        hog_feature = hog.compute(resized_image).flatten()

        # Load the trained SVM model
        svm = joblib.load('TrainData/hog_svm_model.pkl')

        # Predict
        hog_feature = hog_feature.reshape(1, -1)
        prediction = svm.predict(hog_feature)
        predicted_label = prediction[0]

        # Map the predicted label back to the original label
        train_images, train_labels = load_2d_dataset('TrainData/2D/')
        label_to_int = {label: idx for idx, label in enumerate(np.unique(train_labels))}
        int_to_label = {idx: label for label, idx in label_to_int.items()}
        predicted_label_name = int_to_label[predicted_label]

        print(f'Predicted label: {predicted_label_name}')


# Function to convert 2D images to 3D
def convert_to_3d():
    input_dir = 'TrainData/2D/'
    output_dir = 'TrainData/3D_G/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

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
                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # Create a mask using depth information
                    depth_threshold = 1000  # Adjust this threshold based on your depth range
                    mask = np.where((depth_image > 0) & (depth_image < depth_threshold), 255, 0).astype(np.uint8)
                    mask = cv2.resize(mask,
                                      (color_image.shape[1], color_image.shape[0]))  # Resize mask to match color image

                    # Find contours and create a mask for the largest contour
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        mask = np.zeros_like(mask)
                        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                    # Apply the mask to the color image
                    color_image = cv2.bitwise_and(color_image, color_image, mask=mask)

                    # Save the 3D image
                    output_image_path = os.path.join(output_label_path, os.path.splitext(image_name)[0] + '.jpg')
                    cv2.imwrite(output_image_path, color_image)

    pipeline.stop()
    print("Conversion to 3D completed.")


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

convert_button = tk.Button(root, text="Convert 3D", command=convert_to_3d)
convert_button.pack()

root.mainloop()

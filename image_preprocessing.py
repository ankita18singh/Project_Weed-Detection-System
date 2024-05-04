import cv2
import numpy as np
import os

# Define paths to training and testing directories
train_dir = 'train'
test_dir = 'test'
saved_data_dir = 'preprocessed_data'  # Directory to save preprocessed data

# Define image dimensions after resizing
img_height, img_width = 224, 224

# Function to preprocess a single image
def preprocess_image(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return None
        # Resize image
        image_resized = cv2.resize(image, (img_height, img_width))
        # Normalize pixel values
        image_normalized = image_resized.astype(np.float32) / 255.0
        return image_normalized
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

# Function to preprocess images in a directory
def preprocess_images(directory):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = preprocess_image(image_path)
            if image is not None:
                # Append preprocessed image
                images.append(image)
                # Append label based on directory
                labels.append(1 if directory == train_dir else 0)

    return np.array(images), np.array(labels)

# Preprocess training and testing images
train_images, train_labels = preprocess_images(train_dir)
test_images, test_labels = preprocess_images(test_dir)

# Save preprocessed data
np.save(os.path.join(saved_data_dir, 'train_images.npy'), train_images)
np.save(os.path.join(saved_data_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(saved_data_dir, 'test_images.npy'), test_images)
np.save(os.path.join(saved_data_dir, 'test_labels.npy'), test_labels)

print("Preprocessed data saved successfully!")

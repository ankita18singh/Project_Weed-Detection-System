import numpy as np

# Load preprocessed train and test images and labels
train_images = np.load('preprocessed_data/train_images.npy')
test_images = np.load('preprocessed_data/test_images.npy')
train_labels = np.load('preprocessed_data/train_labels.npy')
test_labels = np.load('preprocessed_data/test_labels.npy')

# Save the preprocessed images as features.npy
features_train = train_images.reshape(train_images.shape[0], -1)
features_test = test_images.reshape(test_images.shape[0], -1)
features = np.concatenate((features_train, features_test))
np.save('features.npy', features)

# Save the labels as labels.npy
labels = np.concatenate((train_labels, test_labels))
np.save('labels.npy', labels)
 
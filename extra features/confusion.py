import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Load the extracted features and labels
extracted_features = np.load('features.npy')
labels = np.load('labels.npy')

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(extracted_features, labels, test_size=0.2, random_state=42)

# Define and compile the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(extracted_features.shape[1],)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(features_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(features_test, labels_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict labels for the test set
y_pred = model.predict(features_test)
y_pred = np.round(y_pred).flatten()

# Generate confusion matrix
conf_matrix = confusion_matrix(labels_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Weed', 'Weed'], yticklabels=['No Weed', 'Weed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

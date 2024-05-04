import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_model.keras')

# Load test data
X_test = np.load('test_images.npy')
y_test = np.load('test_labels.npy')

# Make predictions
y_pred = model.predict(X_test)

# Convert probability predictions to binary predictions
y_pred_binary = (y_pred > 0.5).astype(int)

# Define class labels
class_labels = ["No Weed", "Weed"]

# Print classification report with zero_division parameter set to 1
print("Classification Report:")
print(classification_report(y_test, y_pred_binary, target_names=class_labels, zero_division=1))

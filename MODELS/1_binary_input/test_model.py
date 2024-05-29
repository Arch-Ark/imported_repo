import cv2
import tensorflow as tf 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import TrainLogger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import regularizers

from tensorflow.keras.models import load_model
model = load_model('better_model.keras')

# split function
def split_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# load data
# Load the .npz file
test_data = np.load('../reviewed_test_dataset.npz')

test_images = test_data['binary_images']
test_labels = test_data['labels']

# NORMALIZE THE DATA SET
X_test = tf.keras.utils.normalize(test_images, axis = 1)

value_mapping = {14: 10, 16: 11, 23: 12}
y_test = test_labels
for old_value, new_value in value_mapping.items():
    y_test[y_test == old_value] = new_value
print(np.unique(y_test))

# RE SHAPED THE DATASET
X_testr = np.array(X_test).reshape(-1, 28, 28, 1)

# EVALUATING THE TESTING DATA
#test_loss, test_acc = model.evaluate(X_testr, y_test)
#print("Test loss on test samples", test_loss)
#print("Validation Accuracy on test samples", test_acc)

# PREDICT PROBABILITIES
y_pred_prob = model.predict(X_testr)
# CONVERT PROBABILITIES TO CLASS LABELS
y_pred = np.argmax(y_pred_prob, axis=1)

print(len(f'lenth of y_pred{y_pred}'))
print(len(f'lenth of y_test{y_test}'))

# CALCULATE METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# PRINT METRICS
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Assuming y_test contains the true labels and y_pred contains the predicted labels
conf_matrix = confusion_matrix(y_test, y_pred)

# Save confusion matrix to a text file
np.savetxt('2_binary_confusion_matrix.txt', conf_matrix, fmt='%d')

print("Confusion matrix saved to confusion_matrix.txt")
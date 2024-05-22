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
model = load_model('model.keras')

# split function
def split_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# load data
# Load the .npz file
test_data = np.load('../emnist_bymerge_test_dataset.npz')

test_images = test_data['gradient_images']
test_labels = test_data['labels']


X_test = tf.keras.utils.normalize(test_images, axis = 1)
y_test = test_labels

X_testr = np.array(X_test).reshape(-1, 28, 28, 1)



# EVALUATING THE TESTING DATA
test_loss, test_acc = model.evaluate(X_testr, y_test)
print("Test loss on test samples", test_loss)
print("Validation Accuracy on test samples", test_acc)

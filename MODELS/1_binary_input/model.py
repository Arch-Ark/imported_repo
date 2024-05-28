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

# split function
def split_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# load data
# Load the .npz file
train_data = np.load('../reviewed_train_dataset.npz')
test_data = np.load('../reviewed_test_dataset.npz')

# Access the arrays in the .npz file
train_images = train_data['binary_images']
train_labels = train_data['labels']

test_images = test_data['binary_images']
test_labels = test_data['labels']

value_mapping = {14: 10, 16: 11, 23: 12}
y_train = train_labels
for old_value, new_value in value_mapping.items():
    y_train[y_train == old_value] = new_value

y_test = test_labels
for old_value, new_value in value_mapping.items():
    y_test[y_test == old_value] = new_value

# NORMALIZE THE DATA
X_train = train_images.astype('float32') / 255.0
X_test = test_images.astype('float32') / 255.0

# RESHAPE THE IMAGE TO MAKE IT SUITABLE FOR APPLYING CONVOLUTION OPERATION
X_trainr = np.array(X_train).reshape(-1, 28, 28, 1)
X_testr = np.array(X_test).reshape(-1, 28, 28, 1)

# CREATING A DEEP NEURAL NETWORK
model = Sequential()

# Input Layer
model.add(Input(shape = X_trainr.shape[1:]))

# First Convolution Layer
model.add(Conv2D(32, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Layer
model.add(Conv2D(64, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Third Convolution Layer
model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# DropOut
model.add(Dropout(0.3))

# Fully Connected Layers
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(32))
model.add(Activation("relu"))

# CLASSIFICATION LAYER
model.add(Dense(13))
model.add(Activation("softmax"))

# THE SUMMARY OF THE MODEL 
#model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
earlystopper = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, verbose=1)
checkpoint = ModelCheckpoint(f"model.keras", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
trainLogger = TrainLogger('train_log')
tb_callback = TensorBoard(f"logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_accuracy", factor=0.9, min_delta=1e-10, patience=4, verbose=1, mode="auto")

model.fit(
    X_trainr, 
    y_train, 
    epochs=50,
    validation_split= 0.3,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback],
    )

# EVALUATING THE TESTING DATA
#test_loss, test_acc = model.evaluate(X_testr, y_test)
#print("Test loss on test samples", test_loss)
#print("Validation Accuracy on test samples", test_acc)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import TrainLogger

# Split function
def split_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Load data
#train_data = np.load('../emnist_bymerge_train_dataset.npz')
#test_data = np.load('../emnist_bymerge_test_dataset.npz')

# Load the .npz file
train_data = np.load('../reviewed_train_dataset.npz')
test_data = np.load('../reviewed_test_dataset.npz')

# INPUT PIPELINE FOR BINARY INPUT
train_bin_images = train_data['binary_images']
train_labels = train_data['labels']
test_bin_images = test_data['binary_images']
test_labels = test_data['labels']

value_mapping = {14: 10, 16: 11, 23: 12}
y_train = train_labels
for old_value, new_value in value_mapping.items():
    y_train[y_train == old_value] = new_value
print(np.unique(y_train))

y_test = test_labels
for old_value, new_value in value_mapping.items():
    y_test[y_test == old_value] = new_value
print(np.unique(y_test))

# Normalize the data
X_train_bin = tf.keras.utils.normalize(train_bin_images, axis=1)
X_test_bin = tf.keras.utils.normalize(test_bin_images, axis=1)

# Reshape the image to make it suitable for applying convolution operation
X_train_bin = np.array(X_train_bin).reshape(-1, 28, 28, 1)
X_test_bin = np.array(X_test_bin).reshape(-1, 28, 28, 1)

# INPUT PIPELINE FOR THINNED IMAGES
train_thin_images = train_data['thin_images']
test_thin_images = test_data['thin_images']

# Normalize the data
X_train_thin = tf.keras.utils.normalize(train_thin_images, axis=1)
X_test_thin = tf.keras.utils.normalize(test_thin_images, axis=1)

# Reshape the image to make it suitable for applying convolution operation
X_train_thin = np.array(X_train_thin).reshape(-1, 28, 28, 1)
X_test_thin = np.array(X_test_thin).reshape(-1, 28, 28, 1)

# INPUT PIPELINE FOR GRADIENT IMAGES
train_grad_images = train_data['gradient_images']
test_grad_images = test_data['gradient_images']

# Normalize the data
X_train_grad = tf.keras.utils.normalize(train_grad_images, axis=1)
X_test_grad = tf.keras.utils.normalize(test_grad_images, axis=1)

# Reshape the image to make it suitable for applying convolution operation
X_train_grad = np.array(X_train_grad).reshape(-1, 28, 28, 1)
X_test_grad = np.array(X_test_grad).reshape(-1, 28, 28, 1)

# Define input layers for each stream
input_bin = Input(shape=(28, 28, 1))
input_thin = Input(shape=(28, 28, 1))
input_grad = Input(shape=(28, 28, 1))

# Define convolutional layers for binarized images
x_bin = Conv2D(32, (3, 3), activation='relu')(input_bin)
x_bin = MaxPooling2D((2, 2))(x_bin)
x_bin = Conv2D(64, (3, 3), activation='relu')(x_bin)
x_bin = MaxPooling2D((2, 2))(x_bin)
x_bin = Conv2D(128, (3, 3), activation='relu' )(x_bin)
x_bin = MaxPooling2D((2, 2))(x_bin)
x_bin = Flatten()(x_bin)
x_bin = Dropout(0.3)(x_bin)
x_bin = Dense(128, activation='relu')(x_bin)

# Define convolutional layers for thinned images
x_thin = Conv2D(32, (3, 3), activation='relu' )(input_thin)
x_thin = MaxPooling2D((2, 2))(x_thin)
x_thin = Conv2D(64, (3, 3), activation='relu' )(x_thin)
x_thin = MaxPooling2D((2, 2))(x_thin)
x_thin = Conv2D(128, (3, 3), activation='relu' )(x_thin)
x_thin = MaxPooling2D((2, 2))(x_thin)
x_thin = Flatten()(x_thin)
x_thin = Dropout(0.3)(x_thin)
x_thin = Dense(128, activation='relu')(x_thin)

# Define convolutional layers for gradient images
x_grad = Conv2D(32, (3, 3), activation='relu' )(input_grad)
x_grad = MaxPooling2D((2, 2))(x_grad)
x_grad = Conv2D(64, (5, 5), activation='relu' )(x_grad)
x_grad = MaxPooling2D((2, 2))(x_grad)
x_grad = Conv2D(128, (3, 3), activation='relu' )(x_grad)
x_grad = MaxPooling2D((2, 2))(x_grad)
x_grad = Flatten()(x_grad)
x_grad = Dropout(0.3)(x_grad)
x_grad = Dense(128, activation='relu')(x_grad)


############################################################
combined = concatenate([x_bin, x_grad, x_thin])

# Shared dense layers
z = Dense(128, activation='relu')(combined)
z = Dense(64, activation='relu')(z)
output = Dense(13, activation='softmax')(z) 

# Model definition
model = Model(inputs=[input_bin, input_grad, input_thin], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.summary()
#print('MODEL HAS BEEN SUMMARIZED')

# Define callbacks
earlystopper = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, verbose=1)
checkpoint = ModelCheckpoint(f"model.keras", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
trainLogger = TrainLogger('train_log')
tb_callback = TensorBoard(f"logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_accuracy", factor=0.9, min_delta=1e-10, patience=4, verbose=1, mode="auto")

X_train = [X_train_bin, X_train_grad, X_train_thin]

# Model training
model.fit(
    X_train, 
    y_train, 
    epochs=50,
    validation_split=0.3,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback]
)
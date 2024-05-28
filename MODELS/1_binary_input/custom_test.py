import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
model = load_model('model0.keras')

test_data = np.load('../reviewed_test_dataset.npz')
test_labels = test_data['labels']

test_index = 1
test_image = test_data['binary_images'][test_index]

value_mapping = {14: 10, 16: 11, 23: 12}
y_test = test_labels
for old_value, new_value in value_mapping.items():
    y_test[y_test == old_value] = new_value

# NORMALIZE
test_image = tf.keras.utils.normalize(test_image)

# RESHAPE
test_image = test_image.reshape(-1, 28, 28, 1)

predictions = model.predict(test_image)

print(y_test[test_index])
print(predictions)
print(predictions[0].max())

#label = np.argmax(test_image, axis=1)

#print(label[0])

print('###############################################################\n')
# BINARIZE THE images
def binarize_image(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

loaded_arrays = np.load('../custom_scores.npz')
scores = []
for i in loaded_arrays:
    scores.append(loaded_arrays[i])

first = scores[0]
second = scores[1]
third = scores[2]
fourth = scores[3]
fifth = scores[4]
sixth = scores[5]

test_digit = binarize_image(fourth[0].reshape(28,28))

########################################################################
# Rotate the image
test_digit = cv2.rotate(test_digit, cv2.ROTATE_90_CLOCKWISE)
#########################################################################

test_digit = tf.keras.utils.normalize(test_digit)
test_digit = test_digit.reshape(-1, 28, 28, 1)

pred = model.predict(test_digit)
print(pred[0])
print(pred[0].max())
#label = np.argmax(test_digit)

#print(label)

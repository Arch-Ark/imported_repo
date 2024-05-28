import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

# CENTER CHARACTER
def center_pad_image(img, size=28, pad=4):
    # note that the image expected is such that the foreground is black, and the background is white
    #image = 255 - img
    image = img
    # Apply threshold to get a binary image
    _, binary = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Calculate the center of the white region
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate the offset to center the white region
    offset_x = (image.shape[1] // 2) - center_x
    offset_y = (image.shape[0] // 2) - center_y
    
    # Shift the image to center the white region
    centered_image = np.roll(image, offset_x, axis=1)
    centered_image = np.roll(centered_image, offset_y, axis=0)

    # RESIZE THE IMAGE TO (28,28)
    centered_image = cv2.resize(centered_image, (size,size))

    ##### PADDING THE IMAGE
    # Define the amount of padding you want to add
    top_pad = pad
    bottom_pad = pad
    left_pad = pad
    right_pad = pad

    image = centered_image
    # Pad the image
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Display the padded image
    #cv2.imshow('Padded Image', padded_image)
    
    # Resize the padded image to (64, 64) without cropping
    resized_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_NEAREST)
    
    # Display the resized image
    #cv2.imshow('Resized Image', resized_image)
    
    return resized_image


model = load_model('model.keras')

test_data = np.load('../reviewed_test_dataset.npz')
test_labels = test_data['labels']

test_index = 1
test_image = test_data['binary_images'][test_index]

value_mapping = {14: 10, 16: 11, 23: 12}
y_test = test_labels
for old_value, new_value in value_mapping.items():
    y_test[y_test == old_value] = new_value

# NORMALIZE
#test_image = tf.keras.utils.normalize(test_image)
test_image = test_image.astype('float32')/255.0
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

test_digit = binarize_image(sixth.reshape(28,28))

########################################################################
# flip
test_digit = cv2.flip(test_digit, 1)

# rotate
test_digit = cv2.rotate(test_digit, cv2.ROTATE_90_COUNTERCLOCKWISE)
#########################################################################

test_digit = center_pad_image(test_digit)

test_digit = tf.keras.utils.normalize(test_digit)
test_digit = test_digit.reshape(-1, 28, 28, 1)

pred = model.predict(test_digit)
print(pred[0])
print(pred[0].max())
#label = np.argmax(test_digit)

#print(label)

import cv2
import numpy as np

loaded_arrays = np.load('./students_scripts/student3_matric.npz')

for arr in loaded_arrays:
    print(arr)
    image = loaded_arrays[arr]
    cv2.imwrite(f'{arr}.png', image)
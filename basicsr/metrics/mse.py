import numpy as np

def calculate_mse(img1, img2):
    img1 = img1.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    return np.square(np.subtract(img1, img2)).mean()


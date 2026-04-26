import cv2
import numpy as np
from PIL import Image

def corrupt_pil(img, corruption_type):
    img = np.array(img)

    if corruption_type == "blur":
        img = cv2.GaussianBlur(img, (5, 5), 0)

    elif corruption_type == "noise":
        noise = np.random.normal(0, 25, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)

    elif corruption_type == "dark":
        img = img * 0.5

    elif corruption_type == "rotate":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return Image.fromarray(img.astype(np.uint8))
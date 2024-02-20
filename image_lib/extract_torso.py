import numpy as np
import cv2

def extract_torso(full_image_path, full_image, face_image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)

    hull = cv2.convexHull(np.array([[0, 0], [0, 6016], [4016, 6016], [4016, 0]]))
    region_mask = np.zeros_like(gray_image)
    cv2.fillPoly(region_mask, [hull], 255)

    region = cv2.bitwise_xor(full_image, face_image, mask=region_mask)
    cv2.imwrite(full_image_path.split('.png')[0] + '_torso.png', region)

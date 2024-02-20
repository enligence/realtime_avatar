import cv2
import numpy as np
from change_background import change_background


def get_mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

test_path = 'images/'
expect_path = 'expected_images/'

def doit(image):
	expected_image = cv2.imread(expect_path+image)
	test_image = cv2.imread(test_path + image)
	h, w = expected_image.shape[:2]
	
	test_image = cv2.resize(test_image, (w, h))
	cv2.imwrite(test_path + image, test_image)
	change_background(test_path+image)
	test_image = cv2.imread(test_path + image)
	
	return get_mse(test_image, expected_image)
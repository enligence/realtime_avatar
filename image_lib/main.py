import cv2
from change_background import change_background
from process_image import process_image

try:
        image_path = "./images/pic2.png"
        image = cv2.imread(image_path)

        if image is None:
            raise Exception(f"Failed to read the processed image at {image_path}")

        # change_background(image_path)
        process_image(image, image_path)

except Exception as e:
        print(f"An unexpected error occurred: {e}")
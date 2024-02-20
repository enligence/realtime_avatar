import cv2
import numpy as np
from get_faces_landmarks import get_faces_landmarks
from change_background import change_background

class face_regions_extractor:
    def __init__(self, face_image, path_of_face):
        self.image = face_image
        self.output_image = [np.zeros_like(face_image) for _ in range(3)]
        self.parts = ["_head.png", "_eyes.png", "_chin.png", "_torso.png"]
        self.jaw_range = list(range(1, 16))
        
        self.faces, self.dlib_facelandmark = get_faces_landmarks(face_image)
        self.gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        self.face_path = path_of_face
        
    def write_image(self, indeces, ind):
        hull = cv2.convexHull(np.array(indeces))
        region_mask = np.zeros_like(self.gray)
        cv2.fillPoly(region_mask, [hull], 255)
        
        region = cv2.bitwise_and(self.image, self.image, mask=region_mask)
        self.output_image[ind] += region
        
        output_path = self.face_path.split('_')[0] + self.parts[ind]
        cv2.imwrite(output_path, self.output_image[ind])
        
    def extract_regions(self):
        for face in self.faces:
            face_landmarks = self.dlib_facelandmark(self.gray, face)
            if face_landmarks.part(0).x < 6016:
                val = face_landmarks.part(0).x

        # val is basically to ensure we don't overlap one image with another (i.e maintain 256 width for each image in sprite)
        for face in self.faces:
            face_landmarks = self.dlib_facelandmark(self.gray, face)

            pixel_st_x = face_landmarks.part(0).x - val
            pixel_end_x = pixel_st_x + 6015

            # head
            self.write_image([[pixel_st_x, 0], [pixel_end_x, 0], [pixel_st_x, face_landmarks.part(19).y - 5],
                               [pixel_end_x, face_landmarks.part(19).y - 5]], 0)
            # eyes
            self.write_image([[pixel_st_x, face_landmarks.part(19).y - 6], [pixel_end_x, face_landmarks.part(19).y - 6],
                               [pixel_st_x, face_landmarks.part(29).y + 5], [pixel_end_x, face_landmarks.part(29).y + 5]], 1)

            # chin
            self.write_image([[pixel_st_x, face_landmarks.part(29).y + 5], [pixel_end_x, face_landmarks.part(29).y + 5], 
                              [pixel_st_x, 4016], [pixel_end_x, 4016]], 2)

        for part in self.parts:
            change_background(self.face_path.split('_')[0] + part)
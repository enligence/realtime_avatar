from face_regions_extraction import face_regions_extractor
from get_crop_coordinates import get_crop_coordinates
from extract_torso import extract_torso
from get_faces_landmarks import get_faces_landmarks
from segment_face import segment_face
import cv2
import numpy as np

def process_image(image, image_path):
    h, w = image.shape[:2]
    allFaces, _ = get_faces_landmarks(image)
    face_path = image_path.split('.png')[0] + '_face.png'

    for face in allFaces:
        
        '''Crop image to get more facial region and pass the face to get crop coordinates .'''
        head_top, head_left, head_bottom, head_right = get_crop_coordinates(face, h, w)
        cropped_image = image[head_top:head_bottom, head_left:head_right]
        

        '''Now we have cropped image, we need to segment out head '''
        # resize to celebA dataset size for head-segmentation model to work ...
        resized_image = cv2.resize(cropped_image, (178, 218)) 
        seg_map = segment_face(resized_image)
        
        
        '''Resize this segmented face back to original face size '''
        seg_map = cv2.resize(seg_map, (head_right - head_left, head_bottom - head_top))
        mask = np.zeros((h, w), dtype=np.uint8)
        # At this point we have mask of face of same size as in original image ...
        mask[head_top:head_bottom, head_left:head_right] = seg_map  
        

        '''Save the face image ...'''
        face_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(face_path, face_image)
        
        
        '''Extract out torso from original image using the mask'''
        extract_torso(full_image_path=image_path, full_image=image, face_image=face_image)
        
        '''Extract out eyes, head and chin'''
        face_regions_extractor_obj = face_regions_extractor(face_image, face_path)
        face_regions_extractor_obj.extract_regions()
    
    

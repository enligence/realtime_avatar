import cv2, dlib

def get_faces_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # initialising 68 point model ....
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("./dependencies//shape_predictor_68_face_landmarks.dat")
    faces = hog_face_detector(gray, 0)
    
    return faces, dlib_facelandmark
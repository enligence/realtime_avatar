import cv2 
from PIL import Image
from rembg import remove

def change_background_to_none(file_path):
    src = cv2.imread(file_path, 1) 
    
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY) 
    
    b, g, r = cv2.split(src) 
    rgba = [b, g, r, alpha] 
    dst = cv2.merge(rgba, 4) 
    
    cv2.imwrite(file_path, dst) 

def change_background(file_path):
    input = Image.open(file_path) 
    output = remove(input) 
    output.save(file_path) 
    change_background_to_none(file_path)


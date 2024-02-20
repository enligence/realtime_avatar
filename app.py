"""
Create a gradio based application nthat takes in an image from a webcam or upload an image
It then uses the thin plate spline midel to create a video from theimage  using a driver video
It then extracts the viseme images 
It then uses homomorphy to align the images to remove any head movement by mapping all visemes to a single head pose
"""
import gradio as gr
import numpy as np
import cv2
import os
from PIL import Image
from GazeTracking.gaze_tracking import GazeTracking
import urllib.request as urlreq
import torch
import torch.nn as nn
from SemanticGuidedHumanMatting.inference import single_inference
from torchvision.utils import save_image

import sys
# add current path
cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cdir, "head_segmentation"))

from SemanticGuidedHumanMatting.model.model import HumanMatting, HumanSegment
import head_segmentation.head_segmentation.segmentation_pipeline as seg_pipeline



haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
haarcascade_eye_url="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_eye = "haarcascade_eye.xml"

model = HumanMatting(backbone='resnet50')
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("./SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth", map_location=torch.device('cpu')))
model = model.cpu().eval()
print("Load checkpoint successfully ...")



# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("File downloaded")

if (haarcascade_eye in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_eye.xml, < 1MB
    urlreq.urlretrieve(haarcascade_eye_url, haarcascade_eye)
    print("File downloaded")

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")


gaze = GazeTracking()

def get_face_bbox(img):
    # Assuming img is a numpy array with 0-255 range
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in image
    face_cascade = cv2.CascadeClassifier(haarcascade)
    eye_cascade = cv2.CascadeClassifier(haarcascade_eye)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # find biggest face
    max_area = 0
    x_max, y_max, w_max, h_max = 0, 0, 0, 0
    for face in faces:
        x, y, w, h = face[0:4]
        area = w*h
        if area > max_area:
            max_area = area
            x_max, y_max, w_max, h_max = x,y,w,h
    x,y,w,h = x_max, y_max, w_max, h_max

    # for the biggest face find the eyes
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
        
    eyes = eye_cascade.detectMultiScale(roi_gray)
    return {"face_coords": (x,y,w,h), "eye_coords": eyes, "num_faces": len(faces)}

def get_eye_gaze(img):
    gaze.refresh(img)
    return {
        "center_gaze": gaze.is_center(),
        "left_pupil": gaze.pupil_left_coords(),
        "right_pupil": gaze.pupil_right_coords(),
        "is_blinking": gaze.is_blinking(),
    }

def extract_human(img):
    # Assuming img is a numpy array with 0-255 range

    # extract human from image
    pred_alpha, pred_mask = single_inference(model, img)
    # convert pred alpha to 3 channel by duplicating
    pred_alpha_3 = np.dstack((pred_alpha, pred_alpha, pred_alpha))
    # apply alpha to image
    img = (img*pred_alpha_3).astype('uint8')
    
    # Normalize alpha mask to range 0 to 255 
    alpha = cv2.normalize(pred_alpha, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Stack the alpha channel to create 4-channel RGBA image
    img_with_alpha = np.dstack((img, alpha))

    w = img.shape[1]
    h = img.shape[0]
    pred_mask = pred_mask.reshape(h, w)

    return img_with_alpha, pred_mask

def segment_head(image):
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline()

    #convert open cv image to numpy array
    #image = np.array(image)

    segmentation_map = segmentation_pipeline.predict(image)

    return segmentation_map

# extract head
def extract_head(img, human_rect, face_rect):
    target_w = 512 #178
    target_h = 512 #218
    print("human_rect:", human_rect)
    print("face_rect:", face_rect)
    w = img.shape[1]
    h = img.shape[0]
    print("w:", w, " h:", h)
    head_top = max(0, human_rect[1]-10)
    head_bottom = min(face_rect[1] + int(face_rect[3]*1.5), h)
    head_left = max(0, face_rect[0] - int(face_rect[2]*0.25))
    head_right = min(face_rect[0] + face_rect[2] + int(face_rect[2]*0.25), w)
    w1 = head_right - head_left
    h1 = head_bottom - head_top
    aspect = w1/h1
    print("coordinates:", head_top, head_bottom, head_left, head_right, "aspect:", aspect)
    if aspect > target_w/target_h:
        # increase head_bottom to ensure that it meets the target_w/target_h aspect ratio
        head_bottom = int(head_top + w1/target_w*target_h)
        if head_bottom>h:
            head_top = head_bottom - h
            head_bottom = h

            if head_top<0:
                excess = -head_top
                head_top = 0
                compress = int(excess * target_w/target_h)
                head_left += int(compress/2)
                head_right -= int(compress/2)

    else:
        # increase head right and decreate head left to ensure that it meets the target_w/target_h aspect ratio
        new_right = int(head_left + h1/target_h*target_w)    
        extension = new_right - head_right
        head_right += int(extension/2)
        head_left -= int(extension/2)
        excess = 0
        if head_right>w:
            excess = head_right - w
            head_right = w

        if head_left<0:
            excess += -head_left
            head_left = 0

        if excess:
            #reduce excess from bottom and top
            compress = int(excess * target_h/target_w)
            head_top += int(compress/2)
            head_bottom -= int(compress/2)

    w1 = head_right - head_left
    h1 = head_bottom - head_top

    print("coordinates:", head_top, head_bottom, head_left, head_right)
    # crop the image
    cropped_img = img[head_top:head_bottom, head_left:head_right]
    # resize the image
    resized_img = cv2.resize(cropped_img, (target_w, target_h))

    head_mask = segment_head(resized_img)

    

    # given this head mask, we can now segment the head and torso
    # resize the mask to the original image size
    # use suitable interpolation of transparency to avoid jagged edges
    head_mask = cv2.resize(head_mask, (w1, h1), interpolation=cv2.INTER_NEAREST)
    # create a zero image of size w,h and place the mask at head_left and head_top
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[head_top:head_bottom, head_left:head_right] = head_mask
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     epsilon = 0.01 * cv2.arcLength(cnt, True)  # adjust the constant as needed
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.drawContours(mask, [approx], -1, (255), thickness=cv2.FILLED)
        
    # mask now has smoother edges


    # # apply the mask to the original image
    # result = cv2.bitwise_and(img, img, mask=mask)
    #mask_3 = np.dstack((mask, mask, mask))
    #img = img*mask_3
    # make image transparent by adding alpha channel corresponding to the mask
    alpha = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Stack the alpha channel to create 4-channel RGBA image
    img_with_alpha = np.dstack((img, alpha))
    # mask is of shape hxwx1. change it to hxw
    mask = mask.reshape(h, w)
    # return the image and mask
    return img_with_alpha, mask

def get_bbox_of_visible_region(img):
    """
    Given an image 
    If it has alpha get the bbox of non transparent region
    Alternatively get bbox of non black region
    """
    if img.shape[2] == 3:
        # get non zero pixels
        non_zero = cv2.findNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # get bbox
        x, y, w, h = cv2.boundingRect(non_zero)
        return x, y, w, h
    
    # get alpha channel
    alpha = img[:,:,3]
    # get non zero pixels
    non_zero = cv2.findNonZero(alpha)
    # get bbox
    x, y, w, h = cv2.boundingRect(non_zero)
    return x, y, w, h

def crop_head_and_torso(human_img, head_img):
    # get a crop region to crop head and upper torso
    # we can do that by getting the head bbox from the head image 
    # and human bbox from the human image
    # then get the intersection of the two bboxes but not to go too much left/right/bottom
    # to ensure we have the upper torso
    human_bbox = get_bbox_of_visible_region(human_img)
    head_bbox = get_bbox_of_visible_region(head_img)
    """
    find final bbox to crop:
    - if human x and x+w is not very far from head x and head x+w, use human x, x+w
    else add say 25% on either side of head x, x+w
    - if human y is not very far from head y, use human y
    else use add 25% on bottom of head
    Also add 5% on top of human head
    """
    x, y, w, h = human_bbox
    head_x, head_y, head_w, head_h = head_bbox
    final_x = 0
    final_y = 0
    final_w = 0
    final_h = 0
    # get final x
    if head_w/w < 0.6:
        final_x = x
        final_w = w
    else:
        final_x = max(0, int(head_x - 0.25*head_w))
        final_w = w + 2*(head_x - final_x)
    # get final y
    final_y = max(0, min(head_y, y) - int(head_h*0.05))
    final_h = head_h + int(head_h*0.25)

    return final_x, final_y, final_w, final_h

def overlay_transparent_image(background, human_img):
    # If your input image is not in BGRA format, you can convert it:
    # human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2BGRA)
    
    # Split the human image into B,G,R and Alpha channels
    b, g, r, a = cv2.split(human_img)

    # Create a 3-channel image and an alpha channel from the split
    foreground = cv2.merge((b, g, r))
    alpha = cv2.merge((a, a, a))

    # Resize the background to match the human image
    background = cv2.resize(background, (human_img.shape[1], human_img.shape[0]))

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float)/255

    # Perform the alpha blending
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    out_image = cv2.add(foreground, background)

    return out_image.astype('uint8')

def merge_masks(human_mask, head_mask, threshold):
    result_mask = np.zeros_like(human_mask)

    print("SHAPES: ", head_mask.shape, human_mask.shape, result_mask.shape, np.max(human_mask), np.max(head_mask))
    
    for i in range(human_mask.shape[0]):  # Traverse rows
        human_indices = np.where(human_mask[i, :] > 0)[0]
        head_indices = np.where(head_mask[i, :] > 0)[0]
        # If there are no elements, skip this row
        if human_indices.size == 0 or head_indices.size == 0:
            continue
        # Get leftmost and rightmost indices
        human_left, human_right = human_indices[0], human_indices[-1]
        head_left, head_right = head_indices[0], head_indices[-1]
        # Check the difference
        if (human_left - head_left) < threshold and  (human_right - head_right) < threshold:
            result_mask[i, :] = human_mask[i, :]  # Use the human mask
        else:
            result_mask[i, :] = head_mask[i, :]  # Use the head mask
    return result_mask

def get_head_alpha(human_alpha, head_mask):
    """
    Expand head mask to accomodate head_alpha < 255 around it. If head alpha around is 255 then do not expand
    """
    new_alpha = np.zeros_like(human_alpha)
    diff = human_alpha - head_mask
    bright = (human_alpha==255)
    # find last row from bottom where head mask has all 0s
    last_row = 0
    for i in range(human_alpha.shape[0]-1, -1, -1):
        if np.all(head_mask[i, :]==0):
            last_row = i
            break
    # find first row where diff has a series of at most 5, 255 value
    first_row = 0
    for i in range(human_alpha.shape[0]):
        if np.sum(diff[i, :]==255)>5:
            first_row = i
            break
    print("ROWS:", first_row, last_row)


    
def identity_function(img, bg):
    """
    We need to do the following
    1. Get the largest face in the image
    2. If no face, return blank image
    3. Check if the person is looking straight into the camera and is erect
    4. get the human from the image using the SemanticGuidedHumanMatting model
    5. get the head from the other model
    6. get a crop region to crop head and upper torso
    7. Apply thin plate spline to create a video from the cropped human image
    8. Apply homomorphy to align the images to remove any head movement
    9. Extract the viseme images
    """
    # get face bbox
    face_info = get_face_bbox(img)
    # out_img = np.zeros((400,400,3), dtype=np.uint8)
    # if face_info["num_faces"] != 1:
    #     # print on image error
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     org = (50, 50)
    #     fontScale = 1
    #     color = (255, 0, 0)
    #     thickness = 2
    #     out_img = cv2.putText(out_img, 'No face detected' if face_info["num_faces"]==0 else "multiple faces detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
    #     return out_img
    
    # get eyes
    eye_info = get_eye_gaze(img)
    # check if person is looking straight into the camera
    isHorizontal = abs(face_info["eye_coords"][0][1] - face_info["eye_coords"][1][1]) < 10

    # if not eye_info["center_gaze"] or not isHorizontal:
    #     # print on image error
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     org = (50, 50)
    #     fontScale = 1
    #     color = (255, 0, 0)
    #     thickness = 2
    #     out_img = cv2.putText(out_img, 'Person not straight or looking into the camera', org, font, fontScale, color, thickness, cv2.LINE_AA)
    #     return out_img
    
    # extract human
    human_img, human_mask = extract_human(img)
    # human_alpha is 0-255 but human_mask is 0-1
    human_mask = (human_mask*255).astype('uint8')
    # extract alpha channel
    human_alpha = human_img[:,:,3]

    # create another image by removing transparency and add pure white background
    white_bg_img = cv2.cvtColor(human_img, cv2.COLOR_BGRA2BGR)


    human_bbox = get_bbox_of_visible_region(human_img)

    # extract head
    head_img, head_mask = extract_head(white_bg_img, human_bbox, face_info["face_coords"])
    head_mask = (head_mask*255).astype('uint8')

    get_head_alpha(human_alpha, head_mask)
    smoothed_head_mask = merge_masks(human_mask, head_mask, threshold=40) # threshold must be based on image size.
    print(np.min(human_alpha), np.max(human_alpha))
    print(np.min(human_mask), np.max(human_mask))
    print(np.min(head_mask), np.max(head_mask))
    print(np.min(smoothed_head_mask), np.max(smoothed_head_mask))

    # Apply Gaussian Blur to smooth edges
    #blurred_mask = cv2.GaussianBlur(smoothed_head_mask, (21,21), 0)

    # Normalize between 0 and 1
    #smoothed_head_mask = cv2.normalize(blurred_mask, None, 0, 1, cv2.NORM_MINMAX)


    # Now subtract this mask from the human mask
    torso_mask = human_mask - smoothed_head_mask

    #make binary head mask
    #binary_head_mask  = smoothed_head_mask>0

    #torso_mask = human_mask - (binary_head_mask*255).astype('uint8')

    # Reshape result_mask to match the channel size of the image
    stacked_head_mask = np.stack([smoothed_head_mask/255]*4, axis=-1)
    stacked_torso_mask = np.stack([torso_mask/255]*4, axis=-1)

    torso_img = (human_img * stacked_torso_mask).astype('uint8')
    # Apply mask
    head_img = (human_img * stacked_head_mask).astype('uint8')

    x, y, w, h = crop_head_and_torso(human_img, head_img)

    # crop head and upper torso
    img = human_img[y:y+h, x:x+w]

    # resize bg maintaining aspect ration to the human_img and use it as a background
    bg = cv2.resize(bg, (w, h))
    # use bg as a background for human_img that has a transparency layer
    result = overlay_transparent_image(bg, torso_img)
    result = overlay_transparent_image(result, head_img)

    human_rgb = np.zeros((smoothed_head_mask.shape[0], smoothed_head_mask.shape[1], 3), dtype=np.uint8)


    for i in range(3):
        human_rgb[:,:,i] = human_alpha*(smoothed_head_mask>0)
    
    print(np.min(human_rgb), np.max(human_rgb))

    return head_img, torso_img, human_rgb



iface = gr.Interface(
    fn=identity_function, 
    inputs=["image", "image"], 
    outputs=["image", "image", "image"],
)

iface.launch()
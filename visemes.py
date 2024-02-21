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
import torch
from SemanticGuidedHumanMatting.inference import single_inference

import sys
# add current path
cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cdir, "head_segmentation"))

from SemanticGuidedHumanMatting.model.model import HumanMatting, HumanSegment

model = HumanMatting(backbone='resnet50')
model = torch.nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model.load_state_dict(torch.load("./SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth", map_location=torch.device(device)))
#model = model.cpu().eval()
model = model.to(device).eval()
print("Load checkpoint successfully ...")


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
    
def identity_function(img):
    try:
        # extract human
        human_img, human_mask = extract_human(img)
        # human_alpha is 0-255 but human_mask is 0-1
        human_mask = (human_mask*255).astype('uint8')
        # extract alpha channel
        #human_alpha = human_img[:,:,3]

        return human_mask, human_img
    except Exception as e:
        #do stack trace
        import traceback
        traceback.print_exc()
        print("ERROR in exception: ", e)
        
        return img, img


"""
TODO:
1. get the human_img and use it to generate 21 different visemes as mentioned
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis-viseme

Two ways to do this
1. Use thin plate spline model to use a driver video to copy on this image
Problem, the tiny head shakes in the input causes similar head shakes in output that may be difficult to resolve
We can use homomorphy to align the images to remove any head movement by mapping all visemes to a single head pose
but this may still have issues

2. use a lip sync model to sync the human_img with the audio. Extract the visemes and create a viseme sprite
Each image of the viseme must be same size (you can define the size as per your convenience. Currently 256x200)
1st image will be neutral and rest will be in order as mentioned on Azure link above.
These will be stiched into as ingle image from left to right. So if 
single image is of size 100x100
and ther eare 22 images 1 neutral and 21 visemes
the output is 22*100, 100 = 2200,100 image

Certain lip sync algos take in input video and audio and for each frame of video they lyp sync to the ausio
Usually for interactivity people try to create more animated videos, bt we can create static video with neutral
image repeated in each frame for say as many seconds as the audio.\ and  then do lip sync.

Alternatively if you are using other models you may have to ensure that there are no head movements.

"""

iface = gr.Interface(
    fn=identity_function, 
    inputs=["image"], 
    outputs=["image", "image"],
)

iface.launch(share=True)


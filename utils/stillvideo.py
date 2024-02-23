from PIL import Image
from moviepy.editor import *
import numpy as np

def create_video(input_image_path, output_video_path, duration):
    # open image and get dimensions
    img = Image.open(input_image_path)
    width, height = img.size

    # calculate new dimensions
    new_width = min(256, width)
    new_height = int((new_width / width) * height)

    # resize image and convert to numpy array
    resized_img = img.resize((new_width,new_height), Image.BICUBIC)
    np_image = np.array(resized_img)
      
    # create video clip  
    img_clip = ImageClip(np_image, duration=duration)
    
    # write the video file
    img_clip.write_videofile(output_video_path,fps=24) # assuming fps=24

# example usage
#create_video("input.jpg", "output.mp4", 5)

## Running the repo
- Make a virtual environment
- Install the dependencies from the requirements.txt
- Read the description below to get more clear on what the codebase does.
- To perform extraction, run the following command:
  ```bash
    python main.py
  ```

## Description

#### main.py
- This is the start of the library.
- Put the path of the image for which you want to perfrom extraction here.

#### get_faces_landmarks.py
- Change the path of the `.dat` file according to your structure
- Takes image as argument.
- uses the 68 point model to get the landmarks of the face.
- this is called to get the faces and dlib_landmarks object of the image

#### get_crop_coordinates.py
- Takes the face (given by get_faces_landmarks) and height, width of image as argument.
- Return the points of the region to crop to get more facial region in image

#### process_image.py
- This file is called by `main.py` .
- Takes whole_image and it's path as argument
- Perfroms various oprations and handles all the processes that the image goes through
- More detailed description can be found in the file itself, it is well commented

#### segment_face.py
- Uses the head-segmentation model
- Take image as input (*Note*: Image should be of size 178 x 218)
- Return the mask of the face region

#### face_regions_extraction.py
- Operates on the image consisting of only face.
- Extract out eyes, face and head
- Takes face_image and face_image_path as argument

#### change_background.py
- Changes background of the image.
- Takes image_path as argument.

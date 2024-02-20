First of all,To install the dependencies, run the following command:

```
pip install -r requirements.txt
```
# To generate the avatar Video for any text
To generate the video for any text,run the following command:
```
python video_generator.py
```
It will ask you to enter the text for which you want to generate the avatar video.After entering the text,The output will be stored as result.mp4.

# To align the images
Run the following command:
```
python align_images.py
```
This will read the avatar images stored in visemes directory and align them corresponding to the viseme0,i.e,the still frame,and the output for each viseme will be stored in their respective folders by name of aligned_avatar.png

**Note: If you wish to generate the video using aligned images,change the image path in line 64 in video_generator.py to aligned_avatar.png**
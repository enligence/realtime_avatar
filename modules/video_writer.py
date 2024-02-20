import cv2
from pydub import AudioSegment
from omegaconf import DictConfig
import json
# write a video file of generated audio and avatar

def generate(cfg: DictConfig) -> None:
    # other than the exemplified attributes in code below the cfg can contain the below attributes
    #audioPath: str, avatarSpritePath: str, fps: int, avatarHeight: int, avatarWidth: int, outputHeight: int, outputWidth: int
    # Load the audio file
    audio = AudioSegment.from_file(cfg.outPath)
    # Load the avatar sprite
    avatarSprite = cv2.imread(cfg.avatarSpritePath)
    # Load the timestamps of the visemes
    with open(cfg.visemeJsonPath, 'r') as file:
        visemes = json.load(file)

    """
    using the json timings generate the video by selecting the appropriate sprite for the viseme
    TODO
    """
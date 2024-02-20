import azure.cognitiveservices.speech as speechsdk
import os
import base64
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription="7b21760a5b1b43b48db52c037c357844", region="eastus")
output_file = "output.wav"
audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)

# The language of the voice that speaks.
speech_config.speech_synthesis_voice_name='en-US-AndrewNeural'
def generate_voice(input_text):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config,audio_config=audio_config)

    visemes = []

    def viseme_cb(evt):
                viseme_info = {
                    "offset": evt.audio_offset/10000,
                    "viseme": evt.viseme_id, #TODO: either create new images, or correct map when needed
                }
                visemes.append(viseme_info)

    speech_synthesizer.viseme_received.connect(viseme_cb)
    # Get text from the console and synthesize to the default speaker.
    # print("Enter some text that you want to speak >")
    # text = "That quick beige fox jumped in the air over each thin dog and red goat. Look out, I shout, for he's foiled you again, creating chaos in turn."
    text=input_text
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
        print("Audio file saved: {}".format(output_file))
        # audio_data = speech_synthesis_result.audio_data
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

    # encoded_voice_content = base64.b64encode(audio_data)
    return visemes

if __name__ == "__main__":
    text=input("Please enter the text you want to generate avatar output for : ")
    visemes=generate_voice(text)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output.mp4", fourcc, 60.0, (256, 256))

    last_offset = 0
    for item in visemes:
        # calculate the number of frames to display current viseme
        # print(item['offset'])
        # print(last_offset)
        frames = int((item['offset'] - last_offset) / 16.66) # 16.66 ms per frame at 60 fps
        # print("Number of frames are : ",frames)
        # Load the corresponding image for the viseme
        viseme_id=item['viseme']
        if viseme_id==8:
             viseme_id=9
        img = cv2.imread('visemes/viseme{}/avatar.png'.format(viseme_id))

        # Write the image to the video file for 'frames' number of frames
        for _ in range(frames):
            video_writer.write(img)

        last_offset = item['offset']

    # Release everything when the job is finished
    video_writer.release()
    # Load video and audio using moviepy
    video_clip = VideoFileClip("output.mp4")
    audio_clip = AudioFileClip("output.wav")

    # Combine 
    video_clip = video_clip.set_audio(audio_clip)

    # Save the result to a file
    video_clip.write_videofile("result.mp4", codec='mpeg4')
    



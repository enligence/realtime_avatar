import os
from azure.cognitiveservices.speech import (
    SpeechSynthesizer,
    SpeechSynthesisWordBoundaryEventArgs,
    SpeechSynthesisVisemeEventArgs,
)
import azure.cognitiveservices.speech as speechsdk
import base64
import logging
import html
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AzureTTS:
    """
    Used resource:
    https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/python/console/speech_synthesis_sample.py
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureTTS, cls).__new__(cls)
            cls.speech_config = speechsdk.SpeechConfig(
                subscription=os.getenv("AZURE_SPEECH_KEY"),
                region=os.getenv("AZURE_SPEECH_REGION"),
            )
        return cls._instance

    async def generate_voice(
        self, text, voice_name="en-US-GuyNeural", language="en", style="newscast"
    ):
        text = html.escape(text)

        synthesizer = SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )

        visemes = []
        boundaries = []

        def word_boundary_cb(evt: SpeechSynthesisWordBoundaryEventArgs):
            boundaries.append(
                {
                    "text": evt.text,
                    "offset": evt.audio_offset / 10000,
                    "duration": evt.duration / 10000,
                    "text_offset": evt.text_offset,
                    "word_length": evt.word_length,
                }
            )

        def viseme_cb(evt: SpeechSynthesisVisemeEventArgs):
            viseme_info = {
                "offset": evt.audio_offset / 10000,
                "viseme": evt.viseme_id,
            }
            visemes.append(viseme_info)

        synthesizer.viseme_received.connect(viseme_cb)
        synthesizer.synthesis_word_boundary.connect(word_boundary_cb)

        ssml_text = f"<speak version='1.0' xmlns='https://www.w3.org/2001/10/synthesis' xml:lang='{language}'> <voice name='{voice_name}' style='{style}'>{text}</voice></speak>"
        # logger.error(f">>>>>>>>>>>ssml_text: {ssml_text}")
        result = synthesizer.speak_ssml_async(ssml_text).get()
        # print(result)
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # print("Speech synthesized for text [{}]".format(text))
            audio_data = result.audio_data
            # print("{} bytes of audio data received.".format(len(audio_data)))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            # print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                raise ValueError(
                    "Error details: {}".format(cancellation_details.error_details)
                )
        # print("Totally {} bytes received.".format(stream_callback.get_audio_size()))

        valid_visemes_count = len([x for x in visemes if x["viseme"]])
        ratio = valid_visemes_count / len(visemes) if len(visemes) > 0 else 0

        if ratio < 0.5:
            if language.startswith("en"):
                # this is an error
                raise ValueError("Azure did not return visemes")
            else:
                if len(boundaries) == 0:
                    raise ValueError("Azure did not return boundaries")
                # generate visemes using espeak backend
                raise ValueError("Azure did not return visemes")

        encoded_voice_content = base64.b64encode(audio_data)
        del synthesizer

        return visemes, encoded_voice_content.decode("utf-8"), audio_data
    

"""
How to call
from tts.azure import AzureTTS
azure_tts = AzureTTS()
visemes, voice, audio_data = await azure_tts.generate_voice("Hello world")
# visemes is an array of viseme and the offset oi.e. time in ms when that viseme happened in the audio
[{'offset': 50.0, 'viseme': 0},
 {'offset': 125.0, 'viseme': 12},
 {'offset': 225.0, 'viseme': 4},
 {'offset': 287.5, 'viseme': 14},
 {'offset': 375.0, 'viseme': 8},
 {'offset': 475.0, 'viseme': 4},]

voice is the base64 encoded audio
audio_data is the binary audio data

You can save the audio_data to a file and play it using any audio player
with open("audio.wav", "wb") as f:
    f.write(audio_data)

Since this method is async and if you want to call it from a non async method
you can use a

import asyncio
visemes, voice, audio_data = asyncio.run(azure_tts.generate_voice("Hello world"))

For this to work you need to use .env file in project root and add the following
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_speech_region
"""

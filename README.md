# Realtime Avatar

## Installation
We are using poetry to manage the packages



Realtime Avatar that uses sprites and plays them on the fly. It may use transforms to apply head motions.

Inputs -> driver video + avatar image
Outputs -> avatar sprite + sample video output, may be of the same sentence that the driver video is speaking but from generated voice from Azure and not the driver video sound.
In this process, you can write multiple elements that we can pick and chose using hydra based configuration. You can write pipeline (or flow graph) elkements
1. driver video processor
2. avatar image processor
3. viseme mapper
4. drive avatar as per the driver video (can try the two models as per config)
5. extract visemes and generate sprite after homomorphy
6. extract blink sequence, may have to use model to identify eve blink when pupils are in the center. Can use the 68 point model for this as well.
7. overlap eye blink on top of face
8. generate test sentence with random eye blinking in the middle
9. if possible find some metric from say Wav2Lip paper or somewhere else to determine the quality of the output.

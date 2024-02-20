import head_segmentation.segmentation_pipeline as seg_pipeline
import numpy as np

def segment_face(cropped_image):
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline()
    
    image = np.asarray(cropped_image)
    if image.shape[-1] > 3:
        image = image[..., :3]

    segmentation_map = segmentation_pipeline.predict(image)
    return segmentation_map
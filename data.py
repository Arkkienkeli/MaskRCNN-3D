"""
descr here: setting up the data
"""

import os
import skimage.io
import numpy
import config
import utils


class NucleiConfig(config.Config):
    NAME = "nuclei"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + nucleus
    #TRAIN_ROIS_PER_IMAGE = 512
    STEPS_PER_EPOCH = 5000 # check mask_train for the final value
    VALIDATION_STEPS = 50
    #DETECTION_MAX_INSTANCES = 512
    DETECTION_MIN_CONFIDENCE = 0.5
    #DETECTION_NMS_THRESHOLD = 0.35
    #RPN_NMS_THRESHOLD = 0.55


class NucleiDataset(utils.Dataset):

    def initialize(self, pImagesAndMasks, pAugmentationLevel = 0):
        self.add_class("nuclei", 1, "nucleus")

        imageIndex = 0

        for imageFile, maskFile in pImagesAndMasks.items():
            baseName = os.path.splitext(os.path.basename(imageFile))[0]

            image = skimage.io.imread(imageFile)
            if image.ndim < 2 or image.dtype != numpy.uint8:
                continue

            self.add_image(source="nuclei", image_id=imageIndex, path=imageFile, name=baseName, width=image.shape[1],
                           height=image.shape[0], mask_path=maskFile, augmentation_params=None)
            imageIndex += 1

            # TODO adding augmentation parameters

    #def image_reference(self, image_id):

    #def load_image(self, image_id):

    #def load_mask(self, image_id):
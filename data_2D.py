"""
descr here: setting up the data
"""

import os
import skimage.io
import numpy
import config_2D
import utils_2D


class NucleiConfig(config_2D.Config):
    NAME = "nuclei"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + nucleus
    #TRAIN_ROIS_PER_IMAGE = 512
    STEPS_PER_EPOCH = 5 # check mask_train for the final value
    VALIDATION_STEPS = 50
    #DETECTION_MAX_INSTANCES = 512
    DETECTION_MIN_CONFIDENCE = 0.5
    #DETECTION_NMS_THRESHOLD = 0.35
    #RPN_NMS_THRESHOLD = 0.55


class NucleiDataset(utils_2D.Dataset):

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

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        ref = info["name"]
        augmentation = info["augmentation_params"]

        if augmentation is not None:
            ref + " " + str(augmentation)

        return ref

    def load_image(self, image_id):
        info = self.image_info[image_id]
        imagePath = info["path"]
        augmentation = info["augmentation_params"]

        image = skimage.io.imread(imagePath)
        image = RCNNConvertInputImage(image)

        #if augmentation is not None:
        #    image = image_augmentation.Augment(pImage=image, pAugmentationMap=augmentation, pIsMask=False)

        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        maskPath = info["mask_path"]
        augmentation = info["augmentation_params"]

        mask = skimage.io.imread(maskPath)
        if mask.ndim > 2:
            mask = mask[:,:,0]

        #if augmentation is not None:
        #    mask = image_augmentation.Augment(pImage=mask, pAugmentationMap=augmentation, pIsMask=True)

        count = numpy.max(mask)

        masks = numpy.zeros([mask.shape[0], mask.shape[1], count], dtype=numpy.uint8)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                index = int(mask[y,x]) - 1
                if index >= 0:
                    masks[y,x,index] = 1

        #assign class id 1 to all masks
        class_ids = numpy.array([1 for _ in range(count)])
        return masks, class_ids.astype(numpy.int32)

#*****************************************
#                MISC
#*****************************************

def RCNNConvertInputImage(pImageData):
    if pImageData.ndim < 2:
        raise ValueError("Invalid image")
    elif pImageData.ndim < 3:
        pImageData = skimage.color.gray2rgb(pImageData)
    if pImageData.shape[2] > 3:
        pImageData = pImageData[:, :, :3]

    return pImageData
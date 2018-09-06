"""
Training descr here
"""

import sys
import os
import random
import json
import data
import model_2D

print("Usage", sys.argv[0], "settings.json")

class MrcnnTrain:
    __mParams = {}

    def __init__(self, pParams):
        self.__mParams = pParams

    def Train(self):
        trainDir = os.path.join(os.curdir, self.__mParams["train_dir"])
        inModelPath = os.path.join(os.curdir, self.__mParams["input_model"])
        blankInput = self.__mParams["blank_mrcnn"] == "true"
        outModelPath = os.path.join(os.curdir, self.__mParams["output_model"])

        fixedRandomSeed = None
        maxdim = 1024
        trainToValidationChance = 0.2
        augmentationLevel = 0

        # TODO update from config

        if "image_size" in self.__mParams:
            maxdim = int(self.__mParams["image_size"])

        if "stack_size" in self.__mParams:
            maxstack = int(self.__mParams["stack_size"])

        # for random data split
        rnd = random.Random()
        rnd.seed(fixedRandomSeed)
        trainImagesAndMasks = {}
        validationImagesAndMasks = {}

        # iterate through train set
        imagesDir = os.path.join(trainDir, "images")
        masksDir = os.path.join(trainDir, "masks")

        # splitting train data into train and validation
        imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
        for imageFile in imageFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imagesDir, imageFile)
            maskPath = os.path.join(masksDir, baseName + ".tif")
            if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                continue
            if rnd.random() > trainToValidationChance:
                trainImagesAndMasks[imagePath] = maskPath
            else:
                validationImagesAndMasks[imagePath] = maskPath

        # TODO adding evaluation data into validation

        if len(trainImagesAndMasks) < 1:
            raise ValueError("Empty train image list")

        # WHY?
        # just to be non-empty
        if len(validationImagesAndMasks) < 1:
            for key, value in trainImagesAndMasks.items():
                validationImagesAndMasks[key] = value
                break

        # Training dataset
        dataset_train = data.NucleiDataset()
        dataset_train.initialize(pImagesAndMasks=trainImagesAndMasks, pAugmentationLevel=augmentationLevel)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = data.NucleiDataset()
        dataset_val.initialize(pImagesAndMasks=validationImagesAndMasks, pAugmentationLevel=0)
        dataset_val.prepare()

        print("training images (with augmentation):", dataset_train.num_images)
        print("validation images (with augmentation):", dataset_val.num_images)

        config = data.NucleiConfig()
        config.IMAGE_MAX_DIM = maxdim
        config.IMAGE_MIN_DIM = maxdim
        config.IMAGE_STACK_DIM = maxstack
        config.__init__()
        config.display()

        # TODO show setup

        # TODO Load and display random samples

        # Create model in training mode
        mdl = model_2D.MaskRCNN_3D(mode="training", config=config, model_dir=os.path.dirname(outModelPath))

        if blankInput:
            mdl.load_weights(inModelPath, by_name=True,
                             exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            mdl.load_weights(inModelPath, by_name=True)

        allcount = 0
        for epochgroup in self.__mParams["epoch_groups"]:
            epochs = int(epochgroup["epochs"])
            if epochs < 1:
                continue
            allcount += epochs
            mdl.train(dataset_train,
                      dataset_val,
                      learning_rate=float(epochgroup["learning_rate"]),
                      epochs=allcount,
                      layers=epochgroup["layers"])

#        mdl.keras_model.save_weights(outModelPath)


# read in the config file and call the training
jsn = json.load(open(sys.argv[1]))
trainer = MrcnnTrain(pParams=jsn["train_params"])
trainer.Train()
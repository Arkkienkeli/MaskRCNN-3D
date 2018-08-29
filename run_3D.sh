#!/bin/bash

# set Mask_RCNN's path
export PYTHONPATH=$PYTHONPATH:/home/balassa/Devel/python/MaskRCNN_3D

# TODO
#train = yes
#segment = yes
#eval = yes

# read in the config
if [ -z "$1" ]
    then
        echo "Usage: $0 config_file"
        exit 1
fi

# training
python3 train.py $1
if [ $? -ne 0 ]
then
    echo ERROR: "Error during training"
    exit 1
fi
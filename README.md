# Zaitra Challenge

## Introduction
The assignment is to train a segmentation model to segment clouds from the publicly available [Sentinel-2 Cloud Mask Catalogue](https://zenodo.org/records/4172871).

## Dataset
### General information
The dataset consists of 513 [1022x1022] pixel sub-scenes, annotated to three classes CLEAR, CLOUD, CLOUD_SHADOW, however in this challenge, I will focus on the classes CLEAR and CLOUD and consider CLOUD_SHADOW as CLEAR.

### Dataloading and Augmentations
Since the input of the model should be of size [224x224], we will tile the raw scenes into (overlapping) patches of the desired size. We choose 30px as the amount of overlap, because without any overlap, we would be left with  We apply random geometric augmentations (horizontal/vertical flip, rotation) to the patches to reduce the redundancy of the information created by the overlap between the patches. Using no overlap -> we would loose 125*125 pixels worth of information . Thus, if we use an overlap of 30 pixels, we can reduce the amount of informtation lost and get 25 instead of 16 patches.


## Model
At first, I implemented the DeepLabv3 segmentation model with a MobileNet backbone, because it is a relatively small model that I have trained and deployed on hardware before. However, after some initial tests, I was not satisfies with the results and decided to implement a  UNET. I tried using a pretrained ResNet34 as the encoder but eventually ended up with just a simple UNET from scratch. All three models can be found in the [model.py](scripts/model.py).


##  Training
see [config.yaml](config/config.yaml)
30epochs,batch size 1 (memory), crossentropy loss, adam, no schedule, lrs: [0.1,0.05,0.01,0.005,0.001,0.0001]

choose best model 

## Evaluation and results

show iou, accuracy table and some visualisations


## How to run
notebook

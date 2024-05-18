# Zaitra Challenge

## Introduction

The assignment is to train a segmentation model to segment clouds from the publicly available [Sentinel-2 Cloud Mask Catalogue](https://zenodo.org/records/4172871).

## Dataset
### General information
The dataset consists of 513 [1022x1022] pixel sub-scenes, annotated to three classes CLEAR, CLOUD, CLOUD_SHADOW, however in this challenge, I will focus on the classes CLEAR and CLOUD and consider CLOUD_SHADOW as CLEAR.

### Dataloading and Augmentations
Since the input of the model should be of size [224x224], we will tile the raw scenes into overlapping patches of the desired size. We apply random geometric augmentations (horizontal/vertical flip, rotation) to the patches to reduce the redundancy of the information created by the overlap between the patches.


## Model

I have decided to use the 



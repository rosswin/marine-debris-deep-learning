# marine-debris-deep-learning
The preprocessing, training, and evaluation code associated with Ross Winans's Masters Thesis entitled, "Automatic detection of Hawai'i's shoreline stranded mega-debris using deep learning"

## Overview


## Get the Data:
1. Imagery
- Imagery can be viewed as an 
## Folder Descriptions
1. Installation is done with Docker. Designed for Ubuntu 16.04+ workstations.
2. Preprocessing is done to 1) retile GIS imagery and 2) reformat GIS annotations into training data for the Tensorflow Object Detection API (TFODAPI).
3. Training is done with TFODAPI's model_main.py script. This repository contains the .config files that control each object detection model's hyperparameters, data augmentation options, etc. A .pbtxt format map of class name to class numbers is also included as required by TFODAPI.
4. High-level evaluation statistics such as mAP are provided natively by TFODAPI through Tensorboard and do not require additional code. However, additional statistics such as per-class precision/recall and class confusion matricies required additional code. Further, additional code was required to display object detection model predictions as publication quality plots.

## Sources
- [Dockerfiles adapted from Microsoft's AI for Earth Utils](https://github.com/microsoft/ai4eutils/tree/master/TF_OD_API)
- [generate_tfrecord.py adapted from Dat Tran's Raccoon Detector](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)
- [Training .config files adapted from TFODAPI Sample Configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
- [Model Training was conducted with TFODAPI's model_main.py script](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Training checkpoints were preserved using Microsoft AI4E's copy_checkpoints.py script](https://github.com/microsoft/CameraTraps/blob/d61545751c957a92f763fa2f435f1d5f058ed044/detection/detector_training/copy_checkpoints.py)
- [Per-class stats and confusion matrix plots based on svpino's workflow](https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py)

# marine-debris-deep-learning
The preprocessing, training, and evaluation code associated with Ross Winans's Masters Thesis entitled, "Automatic detection of Hawai'i's shoreline stranded mega-debris using deep learning"

## Study Information
Final Link TBD

## Study Abstract
~TBD~

## Overview
- **1--installation** contains Docker installation files. Native environment is Ubuntu 16.04+. Untested elsewhere.
- **2--preprocessing** contains scripts to 1) retile GIS imagery and 2) reformat GIS annotations into training data for the Tensorflow Object Detection API (TFODAPI).
- **3--training** contains the .config files that control each object detection model's hyperparameters, data augmentation options, etc. Also contains a .pbtxt format map of class name to class numbers as required by TFODAPI. TFODAPI's model_main.py script is the main training routine and not included in this repository (see sources below).
- **4--evaluation** contains scripts to calculate per-class statistics and class confuision matricies. Additionally includes scripts for creating publication quality plots from our model outputs. High-level evaluation statistics such as COCO mAP are provided natively by TFODAPI through Tensorboard and do not require additional code. 

## Get the Data:
1. [Source Imagery (GIS Format) available State of Hawai'i Office of Planning as an Esri REST Service.](http://geodata.hawaii.gov/arcgis/rest/services/SoH_Imagery/Coastal_2015/ImageServer)
2. [Source Annotations available from Hawai'i State Department of Land and Natural Resources (DLNR), Department of Aquatic Resources (DAR)](DLNR.aquatics@hawaii.gov)
3. [Reformatted DL Training Data (imagery+annotation) will be available at LILA BC soon](http://lila.science/)

## References
- [Dockerfiles adapted from Microsoft's AI for Earth Utils](https://github.com/microsoft/ai4eutils/tree/master/TF_OD_API)
- [generate_tfrecord.py adapted from Dat Tran's Raccoon Detector](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)
- [Training .config files adapted from TFODAPI Sample Configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
- [Model Training was conducted with TFODAPI's model_main.py script](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Training checkpoints were preserved using Microsoft AI4E's copy_checkpoints.py script](https://github.com/microsoft/CameraTraps/blob/d61545751c957a92f763fa2f435f1d5f058ed044/detection/detector_training/copy_checkpoints.py)
- [Per-class stats and confusion matrix plots based on svpino's workflow](https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py)

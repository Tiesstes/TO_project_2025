The /nn contains training history information, simple model training code (with latest epoch model weights saved), and evaluation on separate data (on images and videos attached).

The //data_augmented is the latest dataset on which the model has been trained. Dataset prepared in roboflow, it's main disadvantage is quite high resolution (3024x4032). 
The augmentations were: +/-10% hue, 90 degree rotation, circa 1.01% noise. There is a substantial class imbalance addressed in YOLO .yaml configuration file. Dataset needs overall improvement for further use.

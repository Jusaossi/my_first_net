# This is my Unet for segmentation (FALL 2020)
Python 3.7 code that uses Unet to gain teeth segmentation.

# Input files that are needed:
* images and target pairs, that can be 2d-slice of 3d- cube

# The main program is  
teet_net.py
# Load_my_batch is
part of my own dataloder for this task
# MyDatasetLoader is
is own dataloder for this task
# my_albumentaion is
handles the data augmentation task
# my_loss_classes is
folder with class of different loss-functions for training the net
# l1_norm is
folder for different metrics to use to measure how good different choices are working
# runbuilderclass is
basically tool for handle different hyperparameters runs
# Runmanagerclass is
tool for keep trak of results and save them for use of tensorboard or as csv... for later handling
# u_net_versions is
folder for differet class of nn versions (like u-net or gated-unet..)

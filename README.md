# Face Detection Using Adaboost And Classifier Cascades


## Abstract

In this project we attempt to create an architecture that can detect faces utilizing
AdaBoost, skin detection, bootstrapping, and classifier cascades. Adaboost tweaks a
set of weak classifiers until they converge into a strong classifier. Skin detection is used
on an image to remove non-skin pixels which increases the accuracy of our face
detector. Bootstrapping helps facilitate training by identifying false positive subwindows
from larger images in our training set and adding them into the training set. Classifier
cascades are used to improve true face detection accuracy while retaining efficiency by
creating multiple layers of classifiers where each one is stronger than the previous.
These weak classifiers are also known as Haar filters and when given an offset, we can
extract features and determine how good of a classifier it was by using the integral of
the face.

## How to run

1. Run directories.m
2. Run train.m or test.m in 
ComputerVisionProj

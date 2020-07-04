# Adaptive Robust Moving Hands Recognition under Complex Lighting Condition

### Introduction
This repository contains the code of the paper "Adaptive Robust Moving Hands Recognition under Complex Lighting Condition". 

This project provides real-time adaptive hand segmentation methods under complex lighting condition using Kinect V2. The code is implemented in Matlab with Mex function.

The code about the Mex function is writen by Terven J. Cordova D.M., "Kin2. A Kinect 2 Toolbox for MATLAB", Science of Computer Programming (https://github.com/jrterven/Kin2). 

The project consists of two major parts, 

1, novel skin color classification using Linear Discriminant Analysis with Gaussian Mixture Model; 

2, background subtraction with adaptive learning rate.

Both parts are combined together with connected neighbor component and an adaptive look up table to realize real-time segment the moving hand pixels from background.

### Dependencies
Kinect2 SDK. http://www.microsoft.com/en-us/download/details.aspx?id=44561

Visual Studio 2012 or newer compiler

MATLAB 2013a or newer (for Visual Studio 2012 support)

MATLAB 2015b or newer for pointCloudDemo2, which uses MATLAB's built-in pointCloud object

### Citation
If you use the code in your research, please cite as,

Z. Qu, "Adaptive robust moving hands recognition under complex lighting condition," 2017 18th International Conference on Advanced Robotics (ICAR), Hong Kong, 2017, pp. 560-565, doi: 10.1109/ICAR.2017.8023666.

    @inproceedings{bib:ICAR17:Qu,
        author={Z. {Qu}},
        booktitle={2017 18th International Conference on Advanced Robotics (ICAR)}, 
        title={Adaptive robust moving hands recognition under complex lighting condition}, 
        year={2017},
        pages={560-565},}

### Authors
Zhongnan Qu

### Licensing 
gpl 3.0

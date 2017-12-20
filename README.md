# Hand-Segmentation

This project provides real-time adaptive hand segmentation methods under complex lighting condition using Kinect V2. The code is implemented in Matlab with Mex function.

The code about the Mex function is writen by Terven J. Cordova D.M., "Kin2. A Kinect 2 Toolbox for MATLAB", Science of Computer Programming (https://github.com/jrterven/Kin2). 

The project consists of two major parts, 

1, novel skin color classification using Linear Discriminant Analysis with Gaussian Mixture Model; 
2, background subtraction with adaptive learning rate.

Both parts are combined together with connected neighbor component and an adaptive look up table to realize real-time segment the moving hand pixels from background.

Authors: Zhongnan Qu

Licensing: gpl 3.0

# Hand-Segmentation

This project provides real-time adaptive hand segmentation methods under complex lighting condition using Kinect V2. The code is implemented in Matlab with Mex function.

The code about the Mex function is writen by Terven J. Cordova D.M., "Kin2. A Kinect 2 Toolbox for MATLAB", Science of Computer Programming (https://github.com/jrterven/Kin2). 

The project consists of two major parts, 

1, novel skin color classification using Linear Discriminant Analysis with Gaussian Mixture Model; 
2, background subtraction with adaptive learning rate.

Both parts are combined together with connected neighbor component and an adaptive look up table to realize real-time segment the moving hand pixels from background.

# Citation

Z. Qu, "Adaptive robust moving hands recognition under complex lighting condition," 2017 18th International Conference on Advanced Robotics (ICAR), Hong Kong, 2017, pp. 560-565, doi: 10.1109/ICAR.2017.8023666.

Authors: Zhongnan Qu

Licensing: gpl 3.0

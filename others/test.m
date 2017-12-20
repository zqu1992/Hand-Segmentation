
addpath('Mex');
clear all
close all

% Create Kinect 2 object and initialize it
% Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color');

% images sizes
color_width = 1920; color_height = 1080;

% Color image is to big, let's scale it down
colorScale = 1/3;

% Create matrices for the images
color = zeros(color_height,color_width,3,'uint8');
colorCali = zeros(color_height,color_width-384,3,'uint8');
colorCaliScale = zeros(color_height/3,(color_width-384)/3,3,'uint8');


% Chrominance component
CbCr = zeros(256,256);

% Transfer matrix from RGB to YCbCr
TranMat = [0.299 0.587 0.114; -0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];
TranMatChro = [-0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];

% color stream figure
figure, h = imshow(colorCaliScale,[]);
title('Color Source');

% Loop last 5s
k = 0;

while k < 100
    
        % Get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    
    % Before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % Copy data to Matlab matrices        
        color = k2.getColor;
        colorCali = color(:, 221:1756, :);

        % update color figure
        colorCaliScale = imresize(colorCali,colorScale);
        set(h,'CData',colorCaliScale); 
    end
    
    
    k = k + 1;
    pause(0.02)
end


% Close kinect object
k2.delete;

close all;

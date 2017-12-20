

addpath('Mex');
clear all
close all

% Create Kinect 2 object and initialize it
% Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color','depth','infrared');

% images sizes
depthWidth = 512; depthHeight = 424; outOfRange = 4000;
colorWidth = 1920; colorHeight = 1080;

% Color image is to big, let's scale it down
colorScale = 1/3;

% Create matrices for the images
depth = zeros(depthHeight,depthWidth,'uint16');
infrared = zeros(depthHeight,depthWidth,'uint16');
color = zeros(colorHeight,colorWidth,3,'uint8');

depthCali = zeros(depthHeight-64,depthWidth,'uint16');
infraredCali = zeros(depthHeight-64,depthWidth,'uint16');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeight/3,(colorWidth-384)/3,3,'uint8');

% depth stream figure
figure, h1 = imshow(depthCali,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar


% color stream figure
figure, h2 = imshow(colorCaliScale,[]);
title('Color Source');


% infrared stream figure
figure, h3 = imshow(infraredCali);
title('Infrared Source');



k = 0;

while k < 200000
    % Get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    
    % Before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % Copy data to Matlab matrices        
        depth = k2.getDepth;
        color = k2.getColor;
        infrared = k2.getInfrared;
        depthCali = depth(27:386, :);
        infraredCali = infrared(27:386, :);
        colorCali = color(:, 221:1756, :);
             
        
        % update depth figure
        depthCali(depthCali>outOfRange) = outOfRange; % truncate depht
        set(h1,'CData',depthCali); 

        % update color figure
        colorCaliScale = imresize(colorCali,colorScale);
        set(h2,'CData',colorCaliScale); 

        % update infrared figure
        %infrared = imadjust(infrared,[0 0.2],[0.5 1]);
        infraredCali = imadjust(infraredCali,[],[],0.5);
        set(h3,'CData',infraredCali); 

    end
    
    k = k + 1;
  
    pause(0.02)
end

% Close kinect object
k2.delete;

close all;

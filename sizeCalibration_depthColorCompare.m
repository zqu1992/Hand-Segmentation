% Create on Dec, 2016
% @author: zhongnan qu 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- GENERAL PARAMETERS DIFINITION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear the workspace
clear all
% add mex file into path
addpath('Mex');
% close all windows
close all
% set the image sizes
depthWidth = 512; depthHeight = 424; 
colorWidth = 1920; colorHeight = 1080;
% set the out of range threshold of depth data
outOfRange = 4000;
% scale the color image down
% scale factor
colorScale = 1/3;
% the color image size after scaling
colorHeightScaled = colorHeight/3;
colorWidthScaled = (colorWidth-384)/3;
% the number of all pixels of scaled color image
numbColorPixel = colorHeightScaled*colorWidthScaled;
% create matrices of the original source and image source after
% calibration
depth = zeros(depthHeight,depthWidth,'uint16');
infrared = zeros(depthHeight,depthWidth,'uint16');
color = zeros(colorHeight,colorWidth,3,'uint8');
depthCali = zeros(depthHeight-64,depthWidth,'uint16');
infraredCali = zeros(depthHeight-64,depthWidth,'uint16');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');

% define the test area (index) in the color image, which will map to depth image
% later 
[s1,s2] = ind2sub([120 120],1:120*120);
IndexTestColor = sub2ind([colorHeight,colorWidth],s1+880,s2+900);

% close all windows
close all

% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color','depth','infrared');
% depth stream image
figure, hDepth = imshow(depth,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar
% depth stream image after calibration
figure, hDepthCali = imshow(depthCali,[0 outOfRange]);
title('Depth Source after Calibration')
colormap('Jet')
colorbar

% infrared stream image
figure, hInfrared = imshow(infrared);
title('Infrared Source');
% infrared stream image after calibration 
figure, hInfraredCali = imshow(infraredCali);
title('Infrared Source after Calibration');

% color stream image
figure, hColor = imshow(color,[]);
title('Color Source');
% color stream image after calibration and scaling
figure, hColorCali = imshow(colorCaliScale,[]);
title('Color Source after Calibration');


% comparing loop
k = 0;
while k < 100
    % time in
    tic
    % get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;    
    % before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % copy data to Matlab matrices
        % get original depth image
        depth = k2.getDepth;
        % get original color image
        color = k2.getColor;
        % get original infrared image
        infrared = k2.getInfrared;
        % depth image calibration
        depthCali = depth(27:386, :);
        % infrared image calibration
        infraredCali = infrared(27:386, :);
        % color image calibration
        colorCali = color(:, 221:1756, :);
        % set extreme depth data as out of range 
        depth(depth>outOfRange) = outOfRange;
        % update depth image
        set(hDepth,'CData',depth);
        % set extreme depth data as out of range 
        depthCali(depthCali>outOfRange) = outOfRange;
        % update calibrated depth image
        set(hDepthCali,'CData',depthCali);
        % update color image
        set(hColor,'CData',color);
        % color image scaling
        colorCaliScale = imresize(colorCali,colorScale);
        % update scaled color image 
        set(hColorCali,'CData',colorCaliScale);
        % adjust the infrared grey scale
        infrared = imadjust(infrared,[],[],0.5);
        % update infrared image
        set(hInfrared,'CData',infrared);
        % adjust the infrared grey scale
        infraredCali = imadjust(infraredCali,[],[],0.5);
        % update calibrated infrared image
        set(hInfraredCali,'CData',infraredCali);
        % reshape the color data 
        colorResh = reshape(permute(color, [3,1,2]), 3, []);
        % get color data in the test area
        dataTestColor = colorResh(:,IndexTestColor);
        % map color pixels in the test area to depth pixels 
        indexTestDepthTmp = k2.mapColorPoints2Depth([s1'+880,s2'+900]);
        % set non-mapping pixels' color index to default (1,1)
        indexTestDepthTmp(indexTestDepthTmp==0) = 1;
        % convert linear indices to subscripts
        indexTestDepth = sub2ind([depthHeight,depthWidth],indexTestDepthTmp(:,1),indexTestDepthTmp(:,2));
        % round the index to integer
        dataTestInfrared = infrared(round(indexTestDepth));
        % set non-mapping pixels' color data to black
        dataTestInfrared(indexTestDepth==1) = 0;
        % cast the data type
        testColorPlot = uint8(reshape(dataTestColor',120,120,3));
        % reshape the color data
        testInfraredPlot = reshape(dataTestInfrared,120,120);
        % comparing test area in color image and in infrared image
        figure(10)
        subplot(1,2,1)
        imshow(testColorPlot,[]);
        subplot(1,2,2)
        imshow(testInfraredPlot);
    end
    pause(0.02);
    % time out
    toc
    % next round
    k = k+1;
end
% Close kinect object
k2.delete;
% close all windows
close all;

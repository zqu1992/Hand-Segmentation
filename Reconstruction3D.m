% Create on Dec, 2016
% @author: zhongnan qu
% add mex file into path
addpath('Mex');
% close all windows
close all
% create Kinect 2 object and initialize it with 'depth' and 'infrared'
k2 = Kin2('depth','infrared');
% set the image sizes
depthWidth = 512; depthHeight = 424; 
% set the out of range threshold of depth data
outOfRange = 4000;

% create matrices of the original source and image source after
% calibration
depth = zeros(depthHeight,depthWidth,'uint16');
infrared = zeros(depthHeight,depthWidth,'uint16');
depthCali = zeros(depthHeight-64,depthWidth,'uint16');
infraredCali = zeros(depthHeight-64,depthWidth,'uint16');

% construction area
% construction area index
[r3D,c3D] = ind2sub([180,256],(1:180*256)');
r3D = r3D + 26;
% depth stream image
figure, hDepthBGInit = imshow(depthCali,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar
% infrared stream image
figure, hInfraredBGInit = imshow(infraredCali);
title('Infrared Source');

% construction loop
i = 1;
while i <= 100
    % get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    % before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % copy data to Matlab matrices
        % get original depth image
        depth = k2.getDepth;
        % get original infrared image
        infrared = k2.getInfrared;
        % depth image calibration
        depthCali = depth(27:386, :);
        % infrared image calibration
        infraredCali = infrared(27:386, :);
        % set extreme depth data as out of range
        depthCali(depthCali>outOfRange) = outOfRange; 
        % update depth figure
        set(hDepthBGInit,'CData',depthCali); 
        % adjust the infrared grey scale
        infraredCali = imadjust(infraredCali,[],[],0.5);
        % update infrared figure
        set(hInfraredBGInit,'CData',infraredCali);
        % map all depth coordinates to coordinates in camera system
        pointsCamera = k2.mapDepthPoints2Camera([r3D,c3D]);
    end
    % next round
    i = i + 1;
    pause(0.02);
end

% Close kinect object
k2.delete;
% close all windows
close all;

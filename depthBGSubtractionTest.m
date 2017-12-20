%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- GENERAL PARAMETERS DIFINITION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear the workspace
clear all
% add mex file into path
addpath('Mex');
% close all windows
close all
% set the number of frames used for initialization
tDepth = 500;

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

% create matrices of the original source and image source after calibration
depth = zeros(depthHeight,depthWidth,'uint16');
infrared = zeros(depthHeight,depthWidth,'uint16');
color = zeros(colorHeight,colorWidth,3,'uint8');
depthCali = zeros(depthHeight-64,depthWidth,'uint16');
infraredCali = zeros(depthHeight-64,depthWidth,'uint16');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
% create a matrix of foreground RGB source after depth background subtraction
colorFG4Depth = zeros(colorHeightScaled,colorWidthScaled,3);
% create a matrix to gather the background depth data
depthSet = zeros(depthHeight-64, depthWidth, tDepth);
% create a matrix of out of range depth value
depthOutOfRange = zeros(depthHeight-64, depthWidth);

% create Kinect 2 object and initialize it with 'depth' and 'infrared'
k2 = Kin2('depth','infrared');
% depth stream image
figure, hDepthBGInit = imshow(depthCali,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar
% infrared stream image
figure, hInfraredBGInit = imshow(infraredCali);
title('Infrared Source');

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DEPTH BACKGROUND SUBTRACTION INITIALIZATION  ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialization loop
i = 1;
while i <= tDepth
    tic;
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
        % update depth image
        set(hDepthBGInit,'CData',depthCali);
        % adjust the infrared grey scale
        infraredCali = imadjust(infraredCali,[],[],0.5);
        % update infrared image
        set(hInfraredBGInit,'CData',infraredCali);
    end
    % store the depth frame into depthSet
    depthSet(:,:,i) = double(depthCali);
    % next round
    i = i + 1;
    pause(0.02);
    % time out
    toc
end

% Close kinect object
k2.delete;
% close all windows
close all;

% judge if the collected depth data is valid
depthValid = bsxfun(@ne, depthSet, depthOutOfRange);
% calculate the mean of background depth data
depthMean = bsxfun(@rdivide, sum(depthSet,3), sum(depthValid,3));
% calculate the variance of background
depthCovTmp1 = bsxfun(@times, bsxfun(@minus, depthSet, depthMean), depthValid);
depthCovTmp2 = bsxfun(@power, depthCovTmp1, 2);
depthCov = bsxfun(@rdivide, sum(depthCovTmp2,3), sum(depthValid,3));
% calculate the standard deviation of background depth data
depthStandard = sqrt(depthCov);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DEPTH BACKGROUND SUBTRACTION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close all windows
close all
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color','depth','infrared');
% depth stream image
figure, hDepth = imshow(depthCali,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar
% infrared stream image
figure, hInfrared = imshow(infraredCali);
title('Infrared Source');
% color stream image
figure, hColor = imshow(colorCaliScale,[]);
title('Color Source');
% foreground stream image
% figure, hFG = imshow(colorFG4Depth,[]);
% title('Skin Source');

% depth foreground detection loop
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
        depthCali(depthCali>outOfRange) = outOfRange; 
        % update depth image
        set(hDepth,'CData',depthCali);
        % color image scaling
        colorCaliScale = imresize(colorCali,colorScale);
        % update color image
        set(hColor,'CData',colorCaliScale);
        % adjust the infrared grey scale
        infrared = imadjust(infrared,[0 0.2],[0.5 1]);
        infraredCali = imadjust(infraredCali,[],[],0.5);
        % update infrared image
        set(hInfrared,'CData',infraredCali);
    end
    pause(0.02);
    % time out
    toc
    % next round
    k = k+1;
end
% Close kinect object
k2.delete;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- IMAGE POSTPROCESSING ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% time in
tic;
% definition of structure element for dilatation 
% [s1,s2] = ind2sub([11 11],1:121);
% structuralElement = [s1-6;s2-6];
[s1,s2] = ind2sub([5 5],1:25);
structuralElement = [s1-3;s2-3];
%structuralElement = [0;0];

% calculate the difference between observation and mean of background depth
% data
depthFGPixel = abs(bsxfun(@minus, double(depthCali), depthMean));
% find the index of pixels whose difference is greater than threshold
depthFGIndex = find(depthFGPixel>depthStandard*200); 
% convert linear indices to subscripts 
[depthFGRow,depthFGCol] = ind2sub([colorHeightScaled,colorWidthScaled],depthFGIndex);
% simple shifting error correction
depthFGCol = depthFGCol + round(bsxfun(@plus,bsxfun(@rdivide, 18015.57,double(depthCali(depthFGIndex))), -22.072174));
% transposition of subscripts
depthFG = [depthFGRow, depthFGCol]';
% set the background infrared value
depthFGTmp = ones(1,numbColorPixel)*40000;
% assign the foreground infrared value
depthFGTmp(:,depthFGIndex) = infraredCali(depthFGIndex);
% reshape the infrared data and normalize it
depthFG4Depth = reshape(depthFGTmp,colorHeightScaled,colorWidthScaled)/40000;

% dilatation on depth foreground coordinates
colorFG = dilatation(depthFG,structuralElement,colorHeightScaled,colorWidthScaled);
% convert subscripts to linear indices
colorFGIndex = sub2ind([colorHeightScaled,colorWidthScaled], colorFG(1,:),colorFG(2,:));
% reshape the color data
colorResh = double(reshape(permute(colorCaliScale, [3,1,2]), 3, []));
% define a matrix of foreground in color image 
colorFGTmp = zeros(3,numbColorPixel);
% assign the foreground color data
colorFGTmp(:,colorFGIndex) = colorResh(:, colorFGIndex);
% reshape the color data
colorFG4Depth = uint8(reshape(colorFGTmp',colorHeightScaled,colorWidthScaled,3));
% infrared image
figure(5)
subplot(2,2,1)
infraredCaliNew = imadjust(infraredCali,[],[],0.5);
imshow(infraredCaliNew);
title('(a)')
% color image
figure(5)
subplot(2,2,2)
imshow(colorCaliScale,[]);
title('(b)')
% foreground in depth image
figure(5)
subplot(2,2,3)
imshow(depthFG4Depth);
title('(c)')
% foreground in color image
figure(5)
subplot(2,2,4)
imshow(colorFG4Depth,[]);
title('(d)')
% time out
toc
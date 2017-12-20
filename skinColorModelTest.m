% Create on Dec, 2016
% @author: zhongnan qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- GENERAL PARAMETERS DIFINITION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear the workspace
clear all;
% add mex file into path
addpath('Mex');
% close all windows
close all;
% set the image sizes
colorWidth = 1920; colorHeight = 1080;
% the number of frames in skin training data collection
tSkin = 800; 
% the number of frames in non-skin training data collection
tNonSkin = 500;

% scale the color image down
% scale factorcolorScale = 1/3;
colorScale = 1/3;
% the color image size after scaling
colorHeightScaled = colorHeight/3;
colorWidthScaled = (colorWidth-384)/3;
% create matrices of the original source and image source after
% calibration
color = zeros(colorHeight,colorWidth,3,'uint8');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
colorOne = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
colorRealTime = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
% define a look up table of skin on chrominance space
CbCrSkin = zeros(256,256);
% define a look up table of non-skin on chrominance space
CbCrNonSkin = zeros(256,256);
% transfer matrix from RGB to YCbCr
%TranMat = [0.299 0.587 0.114; -0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];
% transfer matrix from RGB to CbCr
TranMatChro = [-0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- SKIN COLOR TRAINING DATA COLLECTION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define the detected rectangular area
startPixel = [171,199];
endPixel = [310,384];
% the height and the width of detected rectangular
endStartRect = endPixel-startPixel+1;
% define a matrix to store the RGB data in the rectangular
colorRect = zeros(endStartRect(1),endStartRect(2),3,'uint8');
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color');
% define a matrix to store the CbCr data in the rectangular
CbCrSkin = zeros(256,256);
% color stream image
figure, hSkin = imshow(colorRect,[]);
title('Skin Color');
% skin training data collection loop
i = 0;
while i < tSkin
    % get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    % before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % copy data to Matlab matrices
        % get original color image
        color = k2.getColor;
        % color image calibration
        colorCali = color(:, 221:1756, :);
        % color image scaling
        colorCaliScale = imresize(colorCali,colorScale);
        % get the color data in the rectangular
        colorRect = colorCaliScale(startPixel(1):endPixel(1),startPixel(2):endPixel(2),:);
        % update color image
        set(hSkin,'CData',colorRect); 
    end
    % reshape the color data of skin color in the rectangular 
    colorReshRect = double(reshape(permute(colorRect, [3,1,2]), 3, []));
    % get the sum of RGB data in each pixel
    colorSumRGB = sum(colorReshRect, 1);
    % find the sum over threshlod, consider as skin
    skinIndex = find(colorSumRGB > 200);
    if ~isempty(skinIndex)
        % get the skin RGB data
        skinRGB = colorReshRect(:, skinIndex);
        % tranformation of skin color data from RGB to CbCr
        skinChro = zeros(2,length(skinIndex));
        skinChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', skinRGB),1) + 128;
        skinChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', skinRGB),1) + 128;
        % round the value
        skinChro = round(skinChro);
        % set the value over limit to limit
        skinChro(skinChro<0) = 0;
        skinChro(skinChro>255) = 255;
        % convert subscripts to linear indices
        statisSkinChro = sub2ind(size(CbCrSkin),skinChro(1,:)+1,skinChro(2,:)+1);
        % get the statistic CbCr data of skin color
        tbl = tabulate(statisSkinChro);
        CbCrSkin(tbl(:,1)) = CbCrSkin(tbl(:,1)) + tbl(:,2);
    end
    % next round
    i = i + 1;
    pause(0.02)
end
% Close kinect object
k2.delete;
% close all windows
close all;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- SKIN COLOR TRAINING DATA COLLECTION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color');
% define a matrix to store the CbCr data
CbCrNonSkin = zeros(256,256);
% color stream image
figure, hNonSkin = imshow(colorCaliScale,[]);
title('Non Skin Color');
% non-skin training data collection loop
i = 0;
while i < tNonSkin
    % get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    % before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % copy data to Matlab matrices 
        % get original color image
        color = k2.getColor;
        % color image calibration
        colorCali = color(:, 221:1756, :);
        % color image scaling
        colorCaliScale = imresize(colorCali,colorScale);
        % update color image
        set(hNonSkin,'CData',colorCaliScale); 
    end
    % reshape the color data of non-skin color
    nonSkinRGB = double(reshape(permute(colorCaliScale, [3,1,2]), 3, []));
    % tranformation of non-skin color data from RGB to CbCr
    nonSkinChro = zeros(2,length(nonSkinRGB));
    nonSkinChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', nonSkinRGB),1) + 128;
    nonSkinChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', nonSkinRGB),1) + 128;
    % round the value
    nonSkinChro = round(nonSkinChro);
    % set the value over limit to limit
    nonSkinChro(nonSkinChro<0) = 0;
    nonSkinChro(nonSkinChro>255) = 255;
    % convert subscripts to linear indices
    statisNonSkinChro = sub2ind(size(CbCrNonSkin),nonSkinChro(1,:)+1,nonSkinChro(2,:)+1);
    % get the statistic CbCr data of skin color
    tbl = tabulate(statisNonSkinChro);
    CbCrNonSkin(tbl(:,1)) = CbCrNonSkin(tbl(:,1)) + tbl(:,2);
    % next round
    i = i + 1;
    pause(0.02)
end
% close kinect object
k2.delete;
% close all windows
close all;

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- ESTIMATE GMM (CbCr) PARAMETERS OF SKIN/NON-SKIN COLOR ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the number of skin color components
KSkin = 3; 
% the number of non-skin color components
KNonSkin =5;
% the total number of skin color observation
sum(sum(CbCrSkin))
% use emGMMKmeans function to obtain the parameters
[piSkin, meanSkin, covSkin] = emGmmKmeans(CbCrSkin,KSkin);
% define the look up table of probability given skin class
pSkin = zeros(256,256);
% the probability of each pixel in each skin color component, 
% N(x(i)|mu(k),sigma(k))
gaussian = zeros(256*256,KSkin);
% calculate the gaussian matrix
for i = 1:size(gaussian,1)
    for j = 1:KSkin
        % convert linear indices to subscripts
        [r,c] = ind2sub(size(pSkin),i);
        % using multi variable normal pdf
        gaussian(i,j) = mvnpdf([r;c], meanSkin(:,j),covSkin(:,:,j));
    end
end
% the weighted probability of each pixel in each skin color component
pSkinK = bsxfun(@times, piSkin, gaussian);
% get the look up table of skin
pSum = sum(pSkinK,2);
pSkin(:) = pSum(:);

% the total number of non-skin color observation
sum(sum(CbCrNonSkin))
% use emGMMKmeans function to obtain the parameters
[piNonSkin, meanNonSkin, covNonSkin] = emGmmKmeans(CbCrNonSkin,KNonSkin);
% define the look up table of probability given non-skin class
pNonSkin = zeros(256,256);
% the probability of each pixel in each non-skin color component, 
% N(x(i)|mu(k),sigma(k))
gaussian = zeros(256*256,KNonSkin);
% calculate the gaussian matrix
for i = 1:size(gaussian,1)
    for j = 1:KNonSkin
        % convert linear indices to subscripts
        [r,c] = ind2sub(size(pNonSkin),i);
        % using multi variable normal pdf
        gaussian(i,j) = mvnpdf([r;c], meanNonSkin(:,j),covNonSkin(:,:,j));
    end
end
% the weighted probability of each pixel in each skin color component
pNonSkinK = bsxfun(@times, piNonSkin, gaussian);
% get the look up table of non-skin
pSum = sum(pNonSkinK,2);
pNonSkin(:) = pSum(:);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- GOT ONE FRAME IMAGE ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color');
% color stream image
figure, hOne = imshow(colorOne,[]);
title('One Frame');
% image loop
i = 0;
while i < 100
    % get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    % before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % copy data to Matlab matrices  
        % get original color image
        color = k2.getColor;
        % color image calibration
        colorCali = color(:, 221:1756, :);
        % color image scaling
        colorOne = imresize(colorCali,colorScale);
        % update color image
        set(hOne,'CData',colorOne); 
    end
    % next round
    i = i + 1;
    pause(0.02)
end
% close kinect object
k2.delete;
% close all windows
close all;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL USING BAYES CRITERION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set the threshold value of considering as skin
ratio = 2;
% pSkin/pNonSkin > ratio -> skin
% if this value in look up table is considered as skin
ifSkin = bsxfun(@gt,bsxfun(@minus,pSkin,ratio*pNonSkin),0);
%ifSkin = bsxfun(@gt,pSkin,ratio);
% time in
tic
% reshape the color data
colorOneRGB = double(reshape(permute(colorOne, [3,1,2]), 3, []));
% transformation of color data from RGB to CbCr
colorOneChro = zeros(2,length(colorOneRGB));
colorOneChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', colorOneRGB)) + 128;
colorOneChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', colorOneRGB)) + 128;
% round the value
colorOneChro = round(colorOneChro);
% set the value over limit to limit
colorOneChro(colorOneChro<0) = 0;
colorOneChro(colorOneChro>255) = 255;
% convert subscripts to linear indices
colorOneChroIndex = sub2ind(size(ifSkin), colorOneChro(1,:)+1, colorOneChro(2,:)+1); 
% judge if each pixel belongs to skin and reshape this judgement mask
skinMaskOne = reshape(ifSkin(colorOneChroIndex),colorHeightScaled,colorWidthScaled);
% cast the data type
colorSkinOne = uint8(bsxfun(@times, double(colorOne), skinMaskOne));
figure(1)
% show the result
imshow(colorSkinOne,[]);
% time out
toc
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL IN REAL TIME ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all windows
close all
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'    
k2 = Kin2('color');
% color stream image
figure, hColor = imshow(colorCaliScale,[]);
title('Color Source');
% skin stream image
figure, hRealTime = imshow(colorRealTime,[]);
title('Real Time Skin Detection');
% detection loop
i = 0;
while i < 1000000
    % time in
    tic;
    % get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    % before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % copy data to Matlab matrices
        % get the original color image
        color = k2.getColor;
        % color image calibration
        colorCali = color(:, 221:1756, :);
        % color image scaling 
        colorCaliScale = imresize(colorCali,colorScale);
        % update color image
        set(hColor,'CData',colorCaliScale); 
    end
    % reshape the color data
    colorRealTimeRGB = double(reshape(permute(colorCaliScale, [3,1,2]), 3, []));
    % transformation of color data from RGB to CbCr
    colorSkinRealTimeChro = zeros(2,length(colorRealTimeRGB));
    colorRealTimeChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', colorRealTimeRGB)) + 128;
    colorRealTimeChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', colorRealTimeRGB)) + 128;
    % round the value
    colorRealTimeChro = round(colorRealTimeChro);
    % set the value over limit to limit
    colorRealTimeChro(colorRealTimeChro<0) = 0;
    colorRealTimeChro(colorRealTimeChro>255) = 255;
    % convert subscripts to linear indices
    colorRealTimeChroIndex = sub2ind(size(ifSkin), colorRealTimeChro(1,:), colorRealTimeChro(2,:)); 
    % judge if each pixel belongs to skin and reshape this judgement mask
    skinMaskRealTime = reshape(ifSkin(colorRealTimeChroIndex),colorHeightScaled,colorWidthScaled);
    % cast the data type
    colorRealTime = uint8(bsxfun(@times, double(colorCaliScale), skinMaskRealTime));
    % show the detection result
    set(hRealTime,'CData',colorRealTime); 
    % next round
    i = i + 1;
    pause(0.02)
    % time out
    toc
end
% close kinect object
k2.delete;
% close all windows
close all;

% Create on Dec, 2016
% @author: zhongnan qu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- GENERAL PARAMETERS DIFINITION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear the workspace
clear all
% add mex file into path
addpath('Mex');
% clear the command window
clc;
% set the dimension of used color space (RGB)
dim = 3;
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
% create matrices of foreground source after background subtraction
colorFGTmp = zeros(3,numbColorPixel);
colorFG = uint8(zeros(colorHeightScaled,colorWidthScaled,3));
% create matrices of the original source and image source after
% calibration
color = zeros(colorHeight,colorWidth,3,'uint8');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- BACKGROUND SUBTRACTION INITIALIZATION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get the mean vector and the covariance matrix of background RGB value
[meanBG, covBG] = AdapBGSubtInit(200);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%---  BACKGROUND SUBTRACTION PARAMETERS DIFINITION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define the detected rectangular area
startPixel = [91,129];
endPixel = [270,384];
% number of the gaussian function in each pixel
G = 4; % number of the gaussian function in each pixel
% intial high covariance
covInitHigh = 100;
% initial low weight
wInitLow = 0.05;
% first learning rate
alpha = 0.12;
% threshold value for background percentage
BGPerc = 0.6;
% threshold value for covariance as same component judgement 
ratioCov = 1;

% mean vector of background subtraction GMM model
meanGMM = zeros(dim,G,numbColorPixel);
% covariance matrix of background subtraction GMM model
covGMM = ones(G,numbColorPixel)*covInitHigh;
% weight of background subtraction GMM model
wGMM = zeros(G,numbColorPixel);
% descend order (components in each pixel) of background subtraction GMM model
orderGMM = repmat((1:G)',1,numbColorPixel); 

% assignment
meanGMM(:,1,:) = reshape(permute(meanBG, [3,1,2]), dim,1,[]);
covGMM(1,:) = reshape(covBG, 1, []);
wGMM(1,:) = 1;
% calculate weight/cov
wCovGMM = bsxfun(@rdivide, wGMM, covGMM); % descend order

% close all windows
close all
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
k2 = Kin2('color');
% color stream image
figure (1)
subplot(1,2,1)
hColor = imshow(colorCaliScale,[]);
title('Color Source');
% foreground stream image using adaptive background subtraction
figure (1)
subplot(1,2,2)
hSkin = imshow(colorFG,[]);
title('Foreground Source');

% the height and the width of detected rectangular 
startEndRect = endPixel-startPixel+1;
rectHeight = startEndRect(1);
% index matrix of detected rectangular
[colorS1,colorS2] = ind2sub(startEndRect,1:rectHeight*startEndRect(2));
colorRect = [colorS1+startPixel(1)-1;colorS2+startPixel(2)-1];
colorRectIndex = sub2ind([colorHeightScaled,colorWidthScaled], colorRect(1,:),colorRect(2,:));
% number of pixels of detected rectangular 
numbColorRect = length(colorRectIndex);

% mean vector of background subtraction GMM model in rectangular
meanGMM4ColorRect = zeros(dim,G,numbColorRect);
% covariance matrix of background subtraction GMM model in rectangular
covGMM4ColorRect = ones(G,numbColorRect)*covInitHigh;
% weight of background subtraction GMM model in rectangular
wGMM4ColorRect = zeros(G,numbColorRect);
% descend order (components in each pixel) of background subtraction GMM
% model in rectangular
orderGMM4ColorRect = repmat((1:G)',1,numbColorRect);

% assignment
meanBGTmp = reshape(permute(meanBG, [3,1,2]), dim,1,[]); 
meanGMM4ColorRect(:,1,:) = meanBGTmp(:,:,colorRectIndex);
covBGTmp = reshape(covBG, 1, []);
covGMM4ColorRect(1,:) = covBGTmp(:,colorRectIndex);
wGMM4ColorRect(1,:) = 1;
% calculate weight/covariance in rectangular
wCovGMM4ColorFG = bsxfun(@rdivide, wGMM4ColorRect, covGMM4ColorRect); 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- ADAPTIVE BACKGROUND SUBTRACTION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% detection loop
k = 1;
while k <= 200000
    % time in
    tic;
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
        % update color figure
        set(hColor,'CData',colorCaliScale); 
    end
    
    % reshape the color data (after scaling) as matrix of 3*numbColorPixel
    colorResh = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
    % get the color data in rectangular
    colorRectRGB = double(colorResh(:, colorRectIndex));

    % find the closest component to the observation, and the correspoding
    % Euclidean distance
    [minBias, minBiasIndex] = min( sum( bsxfun(@power, bsxfun(@minus,colorRectRGB,permute(meanGMM4ColorRect,[1,3,2])),2) ,1),[],3);
    % find linear indices of the closest components
    minGMMIndex = bsxfun(@plus, (0:(numbColorRect-1))*G, minBiasIndex);
    % judge if the closest component matches
    ifMatch = bsxfun(@lt, minBias, ratioCov*3*covGMM4ColorRect(minGMMIndex));
    % define an array
    numbColorFGTmp = 1:numbColorRect;
    % find the indices of matched pixels 
    ifMatchIndex = numbColorFGTmp(ifMatch);
    % find the indices of unmatched pixels
    ifNotMatchIndex = numbColorFGTmp(~ifMatch);
    % define the array, which stores the states of observations if detected
    % as foreground or not 
    ifFG4ColorRect = true(1,numbColorRect);
    % update the unmatched pixels
    if ~isempty(ifNotMatchIndex)
        % find the linear indices of the components need to be replaced 
        replaceIndex = bsxfun(@plus, (ifNotMatchIndex-1)*G, orderGMM4ColorRect(end,ifNotMatchIndex));
        % replace the mean vector with the observation
        meanGMM4ColorRect(:,replaceIndex) = colorRectRGB(:,ifNotMatchIndex);
        % replace the covariance with the initial high covariance
        covGMM4ColorRect(replaceIndex) = covInitHigh;
        % update the weights of each component in unmatched pixels
        wGMM4ColorRect(:,ifNotMatchIndex) = bsxfun(@times, wGMM4ColorRect(:,ifNotMatchIndex),...
            (1+bsxfun(@rdivide,(wGMM4ColorRect(replaceIndex)-wInitLow),(1-wGMM4ColorRect(replaceIndex)))));
        % replace the weight with initial low weight
        wGMM4ColorRect(replaceIndex) = wInitLow;
    end
    % update the matched pixels
    if ~isempty(ifMatchIndex)
        % find the linear indices of the components need to be updated 
        updateIndex = bsxfun(@plus,(ifMatchIndex-1)*G, minBiasIndex(ifMatchIndex));
        % update the weights of each component in matched pixels
        wGMM4ColorRect(:,ifMatchIndex) = wGMM4ColorRect(:,ifMatchIndex)*(1-alpha);
        wGMM4ColorRect(updateIndex) = wGMM4ColorRect(updateIndex) + alpha;
        % extract the observation of matched pixels
        colorFGUpdate = colorRectRGB(:,ifMatchIndex);
        % extract the mean vector of the components need to be updated 
        meanGMM4ColorFGUpdate = meanGMM4ColorRect(:,updateIndex);
        % extract the covariance of the components need to be updated
        covGMM4ColorFGUpdate =covGMM4ColorRect(updateIndex);
        % calculate the Gaussian pdf of the observation given matched
        % component
        gaussianTmp = exp(-1/2*bsxfun(@rdivide, sum(bsxfun(@power, bsxfun(@minus, colorFGUpdate, meanGMM4ColorFGUpdate), 2),1), covGMM4ColorFGUpdate)); 
        gaussian4ColorFGUpdate = bsxfun(@times, (2*pi)^(-dim/2)*bsxfun(@power, covGMM4ColorFGUpdate, -dim/2), gaussianTmp);
        % calculate the second learning rate
        rho = gaussian4ColorFGUpdate*alpha;
        one_rho = 1-rho;
        % update the mean vector with the observation
        meanGMM4ColorRect(:,updateIndex) = bsxfun(@plus, bsxfun(@times, one_rho, meanGMM4ColorRect(:,updateIndex)),...
            bsxfun(@times, rho, colorFGUpdate));
        % update the covariance with the observation
        covGMM4ColorRect(updateIndex) = bsxfun(@plus, bsxfun(@times, one_rho, covGMM4ColorRect(updateIndex)),...
            bsxfun(@times, rho, sum(bsxfun(@power, bsxfun(@minus, colorFGUpdate, meanGMM4ColorRect(:,updateIndex)), 2), 1)/3));
        % update the descend order of the matched pixels
        [~,orderTmp] = sort(wCovGMM4ColorFG(:,ifMatchIndex),'descend'); %
        orderGMM4ColorRect(:,ifMatchIndex) = orderTmp;
        % judge if the observations in the matched pixels are foreground
        for i = 1:length(ifMatchIndex)
            ifMatchIndexTmp = ifMatchIndex(i);
            % find the matched components' order 
            FGGMM = find(orderGMM4ColorRect(:,ifMatchIndexTmp)==minBiasIndex(ifMatchIndexTmp));
            % judge if the observation is considered as foreground
            ifFG4ColorRect(ifMatchIndexTmp) = (FGGMM>1)&&(sum(wGMM4ColorRect(orderGMM4ColorRect(1:FGGMM-1,ifMatchIndexTmp),ifMatchIndexTmp)) >= BGPerc);
        end
    end
    % update the weight/covariance
    wCovGMM4ColorFG = bsxfun(@rdivide, wGMM4ColorRect, covGMM4ColorRect);
    % set all pixels as black
    colorFGTmp = zeros(size(colorFGTmp));
    % assign the foreground pixel data into the matrix
    colorFGTmp(:,colorRectIndex(ifFG4ColorRect)) = colorRectRGB(:,ifFG4ColorRect);
    % cast the data type
    colorFG = uint8(reshape(colorFGTmp',colorHeightScaled,colorWidthScaled,3));
    % update detected foreground image
    set(hSkin,'CData',colorFG);
    % next round
    k = k + 1;
    pause(0.02)
    % time out
    toc
end

% Close kinect object
k2.delete;
% close all windows
close all;

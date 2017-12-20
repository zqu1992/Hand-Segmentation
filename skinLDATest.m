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

% the number of frames in skin training data collection
tHand = 120;
% the number of frames in non-skin training data collection
tSpace = 80;
% the dimension of color space, RGB
dim = 3;
% set the initial max value of LDA criterion (first direction w1)
maxJLDA = -1e10;
% set the initial max value of LDA criterion (second direction w2)
maxJLDAOrth = -1e10;
% get the unit direction vector set
dirVec = [0 0 1]';
% set the increment during the calculation
delta = 2*pi*1/200;
for i = 1:100
    % the number of points in latitude ciecle
    J = ceil(2*pi*sin(pi/100*i)/4/delta)*4;
    for j = 0:J-1
        % transform spherical coordinates to Cartesian  
        [a, b, c] = sph2cart(j*2*pi/J,pi/2-pi/100*i,1);
        dirVec(:,end+1) = [a;b;c];
    end
end

% set the image sizes
colorWidth = 1920; colorHeight = 1080;
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
color = zeros(colorHeight,colorWidth,3,'uint8');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
% test image
colorOne = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
%%% for 2 principal component %%%
% skin color RGB table
tbHandRGB = zeros(256*256*256,1);
% non-skin color RGB table
tbSpaceRGB = zeros(256*256*256,1);

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- SKIN COLOR TRAINING DATA COLLECTION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
% define the detected rectangular area
startPixel = [171,199];
endPixel = [310,384];
% the height and the width of detected rectangular
endStartRect = endPixel-startPixel+1;
% define a matrix to store the color data in the rectangular
colorRect = zeros(endStartRect(1),endStartRect(2),3,'uint8');
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
k2 = Kin2('color');
% color stream image
figure, hHandColor = imshow(colorRect,[]);
title('Hand Color Source');
% skin training data collection loop
i = 0;
while i < tHand
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
        % get the color data in the rectangular
        colorRect = colorCaliScale(startPixel(1):endPixel(1),startPixel(2):endPixel(2),:);
        % update color image
        set(hHandColor,'CData',colorRect); 
    end
    % reshape the color data in the rectangular 
    colorReshRect = double(reshape(permute(colorRect, [3,1,2]), 3, []));
    % get the sum of RGB data in each pixel
    colorSumRGB = sum(colorReshRect, 1);
    % find the sum over threshlod, consider as skin 
    handIndex = find(colorSumRGB > 200);
    if ~isempty(handIndex)
        % get the skin RGB data 
        RectHandRGB = double(colorReshRect(:, handIndex));
        % convert subscripts to linear indices
        statisHandRGB = sub2ind([256,256,256],RectHandRGB(1,:)+1,RectHandRGB(2,:)+1,RectHandRGB(3,:)+1);
        % get the statistic data of skin color
        tbHand = tabulate(statisHandRGB);
        tbHandRGB(tbHand(:,1)) = tbHandRGB(tbHand(:,1)) + tbHand(:,2);
    end
    % next round
    i = i + 1;
    % time out
    toc
    pause(0.02)
end
% close kinect object
k2.delete;
% close all windows
close all;
% get the number of skin color observation 
sum(tbHandRGB)
% find the index of valid data, and remove the invalid data
handRGBIndex = find(tbHandRGB>0);
numbHandRGB = tbHandRGB(handRGBIndex);
% convert linear indices to subscripts
[ind1,ind2,ind3] = ind2sub([256,256,256],handRGBIndex);
% get the all collected skin RGB 
dataHandRGB = [ind1,ind2,ind3];
% calculate the mean vector of skin RGB 
meanHandRGB = (sum(bsxfun(@times,dataHandRGB,numbHandRGB))/sum(numbHandRGB))';
% calculate the skin RGB after minusing the mean
handRGBTmp = bsxfun(@minus, dataHandRGB', meanHandRGB);
% calculate the weighted skin RGB after minusing the mean
handRGBTmp1 = bsxfun(@times,reshape(handRGBTmp,3,[],length(handRGBIndex)),reshape(handRGBTmp,[],3,length(handRGBIndex)));
% calculate the covariance matrix of skin RGB
covHandRGB = sum(bsxfun(@times, handRGBTmp1, reshape(numbHandRGB,1,[],length(handRGBIndex))),3)/sum(numbHandRGB);

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- NON-SKIN COLOR TRAINING DATA COLLECTION ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all windows
close all;
% create Kinect 2 object and initialize it
% available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color');
% color stream image
figure, hSpaceColor = imshow(colorCaliScale,[]);
title('Space Color Source');
% non-skin training data collection loop
i = 0;
while i < tSpace
    % time in
    tic
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
        set(hSpaceColor,'CData',colorCaliScale); 
    end
    % reshape the non-skin color data
    dataSpaceCollection = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
    % convert subscripts to linear indices
    statisSpaceRGB = sub2ind([256,256,256],dataSpaceCollection(1,:)+1,dataSpaceCollection(2,:)+1,dataSpaceCollection(3,:)+1);
    % get the statistic data of non-skin color
    tbSpace = tabulate(statisSpaceRGB);
    tbSpaceRGB(tbSpace(:,1)) = tbSpaceRGB(tbSpace(:,1)) + tbSpace(:,2);
    % next round
    i = i + 1;
    % time out
    toc
    pause(0.02)
end
% close kinect object
k2.delete;
% close all windows
close all;
% get the number of non-skin color observation 
sum(tbSpaceRGB)
% find the index of valid data, and remove the invalid data
spaceRGBIndex = find(tbSpaceRGB>0);
numbSpaceRGB = tbSpaceRGB(spaceRGBIndex);
% convert linear indices to subscripts
[ind1,ind2,ind3] = ind2sub([256,256,256],spaceRGBIndex);
% get the all collected skin RGB 
dataSpaceRGB = [ind1,ind2,ind3]';

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- LDA FINDING BEST SEPARATION DIRECTION  ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the initial best direction vector
bestDir = zeros(3,2);
% define the initial last best direction vector
bestDirLast = zeros(3,2);
% define the mean vector of non-skin color data
meanSpaceRGB = [];
% define the covariance matrix of non-skin color data
covSpaceRGB = [];
% define the number of observation in each (K-means) cluster (non-skin)
numbSpaceCluster = [];
% increasing cluster loop
% cluster increases from 3 to 7
for K = 3:7
    % set the last best direction 
    bestDirLast = bestDir;
    % set the initial value of K-means criterion in last round   
    JKmeansLast = 2E8;
    % set the initial value of K-means criterion in this round
    JKmeans = 1E8;
    % descend sort the non-skin color training data, and get the index
    [~,index] = sort(numbSpaceRGB,'descend');
    % get the index of K largest non-skin color training data (larger data means 
    % more observation on this data)
    index = index(1:K);
    % total number of non-skin color observation 
    N = sum(numbSpaceRGB);
    % the number of different training data
    numb = length(numbSpaceRGB);
    % set the K largest training data as initial clusters' mean
    meanK = dataSpaceRGB(:,index);
    % set the initial covariance to zero
    covK = zeros(dim,dim,K);
    % the observation number of each training data in each cluster , size: K*numb
    sumObser = zeros(K,numb);
    % using K-means to divide the non-skin color data into K clusters
    % K-means loop
    while abs(JKmeansLast - JKmeans) >= 1E-8
        %%% E-step %%%
        % calculate all euclidean distances between each training data and each clusters' mean 
        euc_dist_quad_k = reshape(sum(bsxfun(@minus, reshape(dataSpaceRGB,dim,numb,[]), reshape(meanK,dim,[],K)).^2), numb, K);
        % find the minimum euclidean distance of each training data
        % set this training data belonging to this cluster
        [euc_dist_quad,w] = min(euc_dist_quad_k,[],2);
        % set the last criterion
        JKmeansLast = JKmeans;
        % define an array
        k = 1:K;
        
        %%% M-step %%%
        % assign the sumObser matrix
        sumObser = bsxfun(@times, bsxfun(@eq,w,k), numbSpaceRGB)'; % each pixel in which cluster and has how many observations, K*numb
        % calculate the mean of each cluster
        meanKtmp = sum(bsxfun(@times, dataSpaceRGB', reshape(sumObser',numb,[],K)));
        meanK = bsxfun(@rdivide, reshape(meanKtmp, dim, K), sum(sumObser,2)');
        % calculate the training data after minusing the mean
        zeroMean = bsxfun(@minus, reshape(dataSpaceRGB,dim,numb,[]), reshape(meanK,dim,[],K));
        % calculate the weighted training data after minusing the mean
        zeroMeanTmp = bsxfun(@times, zeroMean, reshape(sumObser',[],numb,K));
        % calculate the covariance 
        % this covariance calculation only suitable for 3 dimension
        for i = 1:K
            covK(1,1,i) = zeroMean(1,:,i)*zeroMeanTmp(1,:,i)';
            covK(2,2,i) = zeroMean(2,:,i)*zeroMeanTmp(2,:,i)';
            covK(3,3,i) = zeroMean(3,:,i)*zeroMeanTmp(3,:,i)';
            covK(1,2,i) = zeroMean(1,:,i)*zeroMeanTmp(2,:,i)';
            covK(2,3,i) = zeroMean(2,:,i)*zeroMeanTmp(3,:,i)';
            covK(1,3,i) = zeroMean(1,:,i)*zeroMeanTmp(3,:,i)';
            covK(3,2,i) = covK(2,3,i);
            covK(3,1,i) = covK(1,3,i);
            covK(2,1,i) = covK(1,2,i);
        end
        covK = bsxfun(@rdivide, covK, reshape(sum(sumObser,2),1,1,K));
        % calculate the new criterion
        JKmeansTmp = sum(covK,3);
        JKmeans = trace(JKmeansTmp);
        % set the calculated mean vectors of K-means to the mean vectors 
        % of non-skin clusters 
        meanSpaceRGB = meanK;
        % set the calculated covariance matrix of K-means to the covariance 
        % matrix of non-skin clusters 
        covSpaceRGB = covK;
    end
    % calculate the number of observation in each cluster
    numbK = sum(sumObser,2);
    % set to the number of non-skin clusters
    numbSpaceCluster = numbK;

    % calculate the with-in class scatter matrix on RGB space
    Sw = bsxfun(@plus, covK, covHandRGB);
    % get the bias between mean vectors of K-means and mean vector of skin
    % color
    meanDiff = bsxfun(@minus, meanK, meanHandRGB);
    % calculate the between class scatter matrix on RGB space
    Sb = bsxfun(@times, reshape(meanDiff,dim,[],K), reshape(meanDiff,[],dim,K)); 
    % set the initial max value of LDA criterion (first direction w1)
    maxJLDA = -1e10;
    % set the initial max value of LDA criterion (second direction w2)
    maxJLDAOrth = -1e10;
    % find unit direction vector loop (for w1)
    for i = 1:length(dirVec)
        % calculate the within class scatter matrix on w1 direction
        SwTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVec(:,i),Sw),1), dirVec(:,i)'),2);
        % calculate the between class scatter matrix on w1 direction
        SbTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVec(:,i),Sb),1), dirVec(:,i)'),2);
        % calculate the LDA criterion on w1 direction
        JLDA = sum(bsxfun(@times, bsxfun(@rdivide, SbTmp, SwTmp), reshape(numbK,1,[],K)),3); 
        % judge if the LDA criterion on this w1 is greater than the max
        % value of LDA criterion
        if (JLDA>maxJLDA)
            % if yes, set the LDA criterion on this w1 to the max value of
            % LDA criterion
            maxJLDA = JLDA;
            % set this w1 to the first best direction
            bestDir(:,1) = dirVec(:,i);
        end
    end
    
    % define the unit vector of e_theta 
    eTheta = zeros(3,1);
    % define the unit vector of e_phi
    ePhi = zeros(3,1);
    % calculate the unit vector of e_theta and the unit vector of e_phi on the w1 point
    sqrtTmp = sqrt(bestDir(1,1)^2+bestDir(2,1)^2);
    eTheta(1) = bestDir(3,1)*bestDir(1,1)/sqrtTmp;
    eTheta(2) = bestDir(3,1)*bestDir(2,1)/sqrtTmp;
    eTheta(3) = -sqrtTmp;
    ePhi(1) = -bestDir(2,1)/sqrtTmp;
    ePhi(2) = bestDir(1,1)/sqrtTmp;
    ePhi(3) = 0;
 
    % find unit direction vector loop (for w2)
    for i = 1:360:2*pi
        % get the unit direction vector (w2) in this round
        dirVecOrth = eTheta*sin(i)+ePhi*cos(i);
        % calculate the within class scatter matrix on w2 direction
        SwOrthTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVecOrth,Sw),1), dirVecOrth'),2);
        % calculate the between class scatter matrix on w2 direction
        SbOrthTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVecOrth,Sb),1), dirVecOrth'),2);
        % calculate the LDA criterion on w2 direction
        JLDAOrth = sum(bsxfun(@times, bsxfun(@rdivide, SbOrthTmp, SwOrthTmp), reshape(numbK,1,[],K)),3);
        % judge if the LDA criterion on this w1 is greater than the max
        % value of LDA criterion
        if (JLDAOrth>maxJLDAOrth)
            % if yes, set the LDA criterion on this w2 to the max value of
            % LDA criterion
            maxJLDAOrth = JLDAOrth;
            % set this w2 to the first best direction
            bestDir(:,2) = dirVecOrth;
        end
    end
    % judge if the best unit direction vectors (w1 and w2) in this round 
    % (k round) equal to the last round (k-1 round)
    if (bestDirLast == bestDir)
        % if yes, jump out of the loop
        break;
    end 
end

% calculate the weight of separation between w1 direction and w2 direction
weightHandLDA = maxJLDA / maxJLDAOrth;
% calculate the mean vector of skin color on W-plane
meanHandLDA = bestDir' * meanHandRGB;
% calculate the covariance matrix of skin color on W-plane
covHandLDA = bestDir' * covHandRGB * bestDir;
% calculate the mean vectors of non-skin color on W-plane
meanSpaceLDA = bestDir' * meanSpaceRGB;
% calculate the covariance matrix of non-skin color on W-plane
covSpaceLDA = zeros(size(meanSpaceLDA,1),size(meanSpaceLDA,1),size(meanSpaceLDA,2));
for i=1:size(meanSpaceLDA,2)
    covSpaceLDA(:,:,i) = bestDir' * covSpaceRGB(:,:,i) * bestDir;
end
% calculate the weight of each non-skin cluster 
piSpaceLDA = numbSpaceCluster'/sum(numbSpaceCluster);
% record the number of clusters
K = length(piSpaceLDA);

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DATA NORMALIZATION ON W-PLANE ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% get the set of points which may cause the extreme points on W-plane %%%
% define the points set
extremePoint = zeros(3,256*256*3);
% assign the points in the RGB space with R = 255 to the set
extremePoint(1,1:65536) = 255;
% assign the points in the RGB space with G = 255 to the set
extremePoint(2,65537:131072) = 255;
% assign the points in the RGB space with B = 255 to the set
extremePoint(3,131073:end) = 255;
% get the other two dimension value of these points
% convert linear indices to subscripts
[s1,s2] = ind2sub([256 256],1:65536);
extremePoint(2:3,1:65536) = [s1-1;s2-1];
extremePoint([1,3],65537:131072) = [s1-1;s2-1];
extremePoint(1:2,131073:end) = [s1-1;s2-1];
% find all possible extreme value on w1 direction and on w2 direction
extrVal = bestDir'*extremePoint;
% find the max value on both direction
maxVal = max(extrVal,[],2);
% find the min value on both direction
minVal = min(extrVal,[],2);
% the min value and the max value must at least equal to 0 (from original 
% point in RGB space)
minVal(minVal>0) = 0;
maxVal(maxVal<0) = 0;
% find the slope of normalization on both direction
slope = 255./(maxVal-minVal);
% get the rotation matrix from RGB to W-plane after normalization
RotaMat = bsxfun(@times, bestDir',slope);
% get the translation matrix from RGB to W-plane after normalization
TranMat = -minVal.*slope;

%%% build the look up table for skin color and non-skin color %%%
% define the look up table of skin color on W-plane
skinProjPlane = zeros(256,256);
% transformation of skin color data from RGB to W-plane
skinProj = bsxfun(@plus,RotaMat*dataHandRGB',TranMat);
% round the value
skinProj = round(skinProj);
% set the value over limit to limit
skinProj(skinProj<0) = 0;
skinProj(skinProj>255) = 255;
% convert subscripts to linear indices
skinProjIndex = sub2ind([256,256],skinProj(1,:)+1,skinProj(2,:)+1);
% assign the observation of each training data to the look up table
skinProjPlane(skinProjIndex) = numbHandRGB;

% define the look up table of non-skin color on W-plane
nonSkinProjPlane = zeros(256,256);
% transformation of non-skin color data from RGB to W-plane
nonSkinProj = bsxfun(@plus,RotaMat*dataSpaceRGB,TranMat);
% round the value
nonSkinProj = round(nonSkinProj);
% set the value over limit to limit
nonSkinProj(nonSkinProj<0) = 0;
nonSkinProj(nonSkinProj>255) = 255;
% convert subscripts to linear indices
nonSkinProjIndex = sub2ind([256,256],nonSkinProj(1,:)+1,nonSkinProj(2,:)+1);
% assign the observation of each training data to the look up table
nonSkinProjPlane(nonSkinProjIndex) = numbSpaceRGB;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- ESTIMATE GMM (W-PLANE) PARAMETERS OF SKIN COLOR ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the number of skin color components 
KSkin = 1; 
% the total number of skin color observation 
sum(sum(skinProjPlane))
% use emGMMKmeans function to obtain the parameters 
[piSkin, meanSkin, covSkin] = emGmmKmeans(skinProjPlane,KSkin);
% define the look up table of probability given skin class 
pSkinLDA = zeros(256,256);
% the probability of each pixel in each skin color component, 
% N(x(i)|mu(k),sigma(k))
gaussian = zeros(256*256,KSkin);
% calculate the gaussian matrix
for i = 1:size(gaussian,1)
    for j = 1:KSkin
        % convert linear indices to subscripts
        [r,c] = ind2sub(size(pSkinLDA),i);
        % in case the overfitting problem
        % judge if the dimension is reduced
        if (det(covSkin(:,:,j))>0)
            % if not, using multi variable normal pdf
            gaussian(i,j) = mvnpdf([r;c],meanSkin(:,j),covSkin(:,:,j));
        % if yes
        % judge which dimension is disappeared
        elseif (covSkin(1,1,j)==0)
            % w1 direction is disapeared, using normal pdf
            gaussian(i,j) = normpdf(c,meanSkin(2,j),covSkin(2,2,j));
        else
            % w2 direction is disapeared, using normal pdf
            gaussian(i,j) = normpdf(r,meanSkin(1,j),covSkin(1,1,j));
        end
    end
end
% the weighted probability of each pixel in each skin color component
pSkinK = bsxfun(@times, piSkin, gaussian);
% get the look up table of skin
pSum = sum(pSkinK,2);
pSkinLDA(:) = pSum(:);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- ESTIMATE GMM (W-PLANE) PARAMETERS OF NON-SKIN COLOR ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the number of non-skin components
KNonSkin = 5;
% the total number of non-skin color observation 
sum(sum(nonSkinProjPlane))
% use emGMMKmeans function to obtain the parameters 
[piNonSkin, meanNonSkin, covNonSkin] = emGmmKmeans(nonSkinProjPlane,KNonSkin);
% define the look up table of probability given non-skin class
pNonSkinLDA = zeros(256,256);
% the probability of each pixel in each non-skin color component, 
% N(x(i)|mu(k),sigma(k))
gaussian = zeros(256*256,KNonSkin);
% calculate the gaussian matrix
for i = 1:size(gaussian,1)
    for j = 1:KNonSkin
        % convert linear indices to subscripts
        [r,c] = ind2sub(size(pNonSkinLDA),i);
        % in case the overfitting problem
        % judge if the dimension is reduced
        if (det(covNonSkin(:,:,j))>0)
            % if not? using multi variable normal pdf
            gaussian(i,j) = mvnpdf([r;c],meanNonSkin(:,j),covNonSkin(:,:,j));
        % if yes
        % judge which dimension is disappeared
        elseif (covNonSkin(1,1,j)==0)
            % w1 direction is disapeared, using normal pdf
            gaussian(i,j) = normpdf(c,meanNonSkin(2,j),covNonSkin(2,2,j));
        else
            % w2 direction is disapeared, using normal pdf
            gaussian(i,j) = normpdf(r,meanNonSkin(1,j),covNonSkin(1,1,j));
        end
    end
end
% the weighted probability of each pixel in each skin color component
pNonSkinK = bsxfun(@times, piNonSkin, gaussian);
% get the look up table of skin
pSum = sum(pNonSkinK,2);
pNonSkinLDA(:) = pSum(:);

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
while i < 200
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL USING BAYES CRITERION (2D) ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set the threshold value of considering as skin
ratio = 4;
% pSkin/pNonSkin > ratio -> skin
% if this value in look up table is considered as skin
ifSkinLDA = bsxfun(@gt,bsxfun(@minus,pSkinLDA,ratio*pNonSkinLDA),0);
% ifSkin = bsxfun(@gt,pSkin,ratio);

% time in
tic
% reshape the color data
colorOneRGB = double(reshape(permute(colorOne, [3,1,2]), 3, []));
% project the RGB color data on W-plane and normalization
colorOneProj = bsxfun(@plus,RotaMat*colorOneRGB,TranMat);
% round the value
colorOneProj = round(colorOneProj);
% set the value over limit to limit
colorOneProj(colorOneProj<0) = 0;
colorOneProj(colorOneProj>255) = 255;
% convert subscripts to linear indices
colorOneProjIndex = sub2ind(size(ifSkinLDA), colorOneProj(1,:)+1, colorOneProj(2,:)+1); 
% judge if each pixel belongs to skin and reshape this judgement mask
skinMaskOne = reshape(ifSkinLDA(colorOneProjIndex),colorHeightScaled,colorWidthScaled);
% cast the data type
colorSkinOne = uint8(bsxfun(@times, double(colorOne), skinMaskOne));
% show the detection result
figure(2)
imshow(colorSkinOne,[]);
title('skin LDA')
% time out
toc


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL USING ELLIPSE (2D) ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define a matrix of weighted covariance 
covHandLDAWeighted = [];
% calculate the value in this weighted covariance matrix
covHandLDAWeighted = covHandLDA;
covHandLDAWeighted(1,1) = covHandLDAWeighted(1,1)*sqrt(weightHandLDA);
covHandLDAWeighted(2,2) = covHandLDAWeighted(1,1)/sqrt(weightHandLDA);
% calculate some parameters
% sigma1*sigma2
sigmaProduct = sqrt(covHandLDAWeighted(1,1)*covHandLDAWeighted(2,2));
% correlation coefficient of both direction
rho = covHandLDAWeighted(1,2)/sigmaProduct;
% 1-rho^2
one_rho2 = 1-rho^2;
% 2*rho/(sigma1*sigma2)
tworho__sigmaProduct = 2*rho/sigmaProduct;
% time out
tic
% reshape the color data
colorOneResh = double(reshape(permute(colorOne, [3,1,2]), 3, []));
% project the RGB color data on W-plane
colorLDA = bestDir'*colorOneResh;
% calculate the bias of observation and mean vector
colorLDABias = bsxfun(@minus, colorLDA, meanHandLDA);
% assign each observation to the ellipse function (weighted)
ellipse1 = sum(bsxfun(@rdivide, bsxfun(@power,colorLDABias,2), diag(covHandLDAWeighted)));
ellipse2 = bsxfun(@times, bsxfun(@times, colorLDABias(1,:),colorLDABias(2,:)), tworho__sigmaProduct);
ellipse = bsxfun(@minus, ellipse1, ellipse2);
% judge if each observation is in the ellipse
ifHandColor = bsxfun(@lt,ellipse,one_rho2);
% get the color data of skin pixels
colorSkinEllipseTmp = zeros(3,numbColorPixel);
colorSkinEllipseTmp(:,ifHandColor) = colorOneResh(:,ifHandColor);
% cast the data type
colorSkinEllipse = uint8(reshape(colorSkinEllipseTmp',colorHeightScaled,colorWidthScaled,3));
% show the detection result
figure (2)
imshow(colorSkinEllipse,[]);
title('color skin in 2D ellipse');
% time out
toc

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL USING BAYES CRITERION (1D) ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set the threshold value of considering as skin
ratio = 1;
% define a flag if the left boundary still needs to shift to the left
leftShift = true;
% define a flag if the right boundary still needs to shift to the right
rightShift = true;
% set mean value to the initial left boundary 
leftBoundary = meanHandLDA(1);
% set mean value to the initial right boundary
rightBoundary = meanHandLDA(1);
% the number of non-skin GMM component 
K = 5;
% left/right boundary finding loop
while(leftShift||rightShift)
    % judge if the left boundary still needs to shift to the left
    if (leftShift)
        % if yes, reduce the left boudary by one step, 0.01
        leftBoundary = leftBoundary-0.01;
        % calculate the probability of this left boundary given skin class
        pSkinLDA = normpdf(leftBoundary,meanHandLDA(1),sqrt(covHandLDA(1,1))); 
        % calculate the probability of this left boundary given non-skin class
        pNonSkinLDA = 0;
        % calculate the sum probability of all non-skin components
        for i=1:K
            pNonSkinLDA = pNonSkinLDA + piSpaceLDA(i)*normpdf(leftBoundary,meanSpaceLDA(1,i),covSpaceLDA(1,1,i));
        end
        % judge if the left boundary still needs to shift to the left
        % pSkin/pNonSkin > ratio -> skin
        if (pSkinLDA<(pNonSkinLDA*ratio))
            % if not, set the flag leftShift to false
            leftShift = false;
        end
    end
    % judge if the right boundary still needs to shift to the right
    if (rightShift)
        % if yes, increase the right boudary by one step, 0.01
        rightBoundary = rightBoundary+0.01;
        pSkinLDA = normpdf(rightBoundary,meanHandLDA(1),sqrt(covHandLDA(1,1))); 
        % calculate the probability of this right boundary given non-skin class
        pNonSkinLDA = 0;
        % calculate the sum probability of all non-skin components
        for i=1:K
            pNonSkinLDA = pNonSkinLDA + piSpaceLDA(i)*normpdf(rightBoundary,meanSpaceLDA(1,i),covSpaceLDA(1,1,i));
        end
        % judge if the right boundary still needs to shift to the right
        % pSkin/pNonSkin > ratio -> skin
        if (pSkinLDA<(pNonSkinLDA*ratio))
            % if not, set the flag rightShift to false
            rightShift = false;
        end
    end
end
% time in
tic
% reshape the color data
colorOneResh = double(reshape(permute(colorOne, [3,1,2]), 3, []));
% project the RGB color data on w1 direction
colorLDA = bestDir(:,1)'*colorOneResh;
% judge if each projected color data locates between both boundary
ifHandColor = bsxfun(@and,bsxfun(@lt, colorLDA, rightBoundary),bsxfun(@gt, colorLDA, leftBoundary));
% get the color data of skin pixels
colorSkin1DTmp = zeros(3,numbColorPixel);
colorSkin1DTmp(:,ifHandColor) = colorOneResh(:,ifHandColor);
% cast the data type
colorSkin1D = uint8(reshape(colorSkin1DTmp',colorHeightScaled,colorWidthScaled,3));
% show the result
figure (2)
imshow(colorSkin1D,[]);
title('color skin in 1D');
% time out
toc


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL USING BAYES CRITERION (3D RGB) ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set the threshold value of considering as skin
ratio = 1;
% define a 3D look up table in RGB space
RGBIfSkin = false(256,256,256);
% R,G,B loop
% R loop
for r = 0:255
    % G loop
    for g = 0:255
        % B loop
        for b = 0:255
            % calculate the projected data of this RGB 
            projRGB = bestDir'*[r;g;b];
            % calculate the probability of this RGB given skin class
            pSkinLDA = mvnpdf(projRGB,meanHandLDA,covHandLDA); 
            % calculate the probability of this RGB given non-skin class
            pNonSkinLDA = 0;
            % calculate the sum probability of all non-skin components
            for i=1:K
                pNonSkinLDA = pNonSkinLDA + piSpaceLDA(i)*mvnpdf(projRGB,meanSpaceLDA(i),covSpaceLDA(:,:,i));
            end
            % judge if this RGB is considered as skin 
            % pSkin/pNonSkin > ratio -> skin
            if (pSkinLDA>(pNonSkinLDA*ratio))
                % if yes, set this RGB to true
                RGBIfSkin(r+1,g+1,b+1) = true;
            end
        end
    end
end
% time in
tic
% reshape the color data
colorOneResh = double(reshape(permute(colorOne, [3,1,2]), 3, []));
% convert subscripts to linear indices
RGBIndex = sub2ind([256,256,256],colorOneResh(1,:),colorOneResh(2,:),colorOneResh(3,:));
% judge if each data is considered as skin
ifHandColor = RGBIfSkin(RGBIndex);
% get the color data of skin pixels
colorSkinTmp = zeros(3,numbColorPixel);
colorSkinTmp(:,ifHandColor) = colorOneResh(:,ifHandColor);
% cast the data type
colorSkin = uint8(reshape(colorSkinTmp',colorHeightScaled,colorWidthScaled,3));
% show the result
figure (2)
imshow(colorSkin,[]);
title('color skin');
% time out
toc


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--- DETECT SKIN PIXEL IN REAL TIME ---%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all windows
close all
% set the threshold value of considering as skin
ratio = 4;
% pSkin/pNonSkin > ratio -> skin
% get the look up table of if this value is considered as skin
ifSkinLDA = bsxfun(@gt,bsxfun(@minus,pSkinLDA,ratio*pNonSkinLDA),0);
% ifSkin = bsxfun(@gt,pSkin,ratio);

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
    colorRGB = double(reshape(permute(colorCaliScale, [3,1,2]), 3, []));
    % project the RGB color data on W-plane and normalization
    colorProj = bsxfun(@plus,RotaMat*colorRGB,TranMat);
    % round the value
    colorProj = round(colorProj);
    % set the value over limit to limit
    colorProj(colorProj<0) = 0;
    colorProj(colorProj>255) = 255;
    % convert subscripts to linear indices
    colorProjIndex = sub2ind(size(ifSkinLDA), colorProj(1,:)+1, colorProj(2,:)+1);
    % judge if each pixel belongs to skin and reshape this judgement mask
    skinMask = reshape(ifSkinLDA(colorProjIndex),colorHeightScaled,colorWidthScaled);
    % cast the data type
    colorRealTime = uint8(bsxfun(@times, double(colorCaliScale), skinMask));
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

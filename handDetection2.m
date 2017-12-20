addpath('Mex');
clc;
%clear all

dim = 3;
modeHandLDA = 2;

% [s1,s2] = ind2sub([11 11],1:121);
% structuralElement = [s1-6;s2-6];

%[s1,s2] = ind2sub([5 5],1:25);
%structuralElement = [s1-3;s2-3];

structuralElement = [0;0];

G = 3; % number of the gaussian function in each pixel 
alpha = 0.1;
covInitHigh = 100;
wInitLow = 0.05;
BGPerc = 0.6;





% images sizes
depthWidth = 512; depthHeight = 424; outOfRange = 4000;
colorWidth = 1920; colorHeight = 1080;

% Color image is to big, let's scale it down
colorScale = 1/3;
colorHeightScaled = colorHeight/3;
colorWidthScaled = (colorWidth-384)/3;
numbColorPixel = colorHeightScaled*colorWidthScaled;
colorSkinTmp = zeros(3,numbColorPixel);
colorSkin = uint8(zeros(colorHeightScaled,colorWidthScaled,3));


%% Hand LDA
pause(5)

[bestDirHandLDA, meanHandLDA, covHandLDA, weightHandLDA] = skinLDA(50,20,modeHandLDA);

if (modeHandLDA==2)
    covHandLDAWeighted = covHandLDA;
    covHandLDAWeighted(1,1) = covHandLDAWeighted(1,1)*sqrt(weightHandLDA);
    covHandLDAWeighted(2,2) = covHandLDAWeighted(1,1)/sqrt(weightHandLDA);
    sigmaProduct = sqrt(covHandLDAWeighted(1,1)*covHandLDAWeighted(2,2));
    rho = covHandLDAWeighted(1,2)/sigmaProduct;
    one_rho2 = 1-rho^2;
    tworho__sigmaProduct = 2*rho/sigmaProduct;
end

pause(3)
%% Depth Background Initialization
[depthMean, depthCov] = depthBackgroundInit(500);
depthStandard = sqrt(depthCov);
depthStandard3 = depthStandard * 3;
depthStandard6 = depthStandard * 6;

%% Background Substraction Initialization
[meanBG, covBG] = BGSubtractInit(500);

%%
close all

structuralElement = [0;0];


% Create Kinect 2 object and initialize it
% Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color','depth');

% Create matrices for the images
depth = zeros(depthHeight,depthWidth,'uint16');
color = zeros(colorHeight,colorWidth,3,'uint8');

depthCali = zeros(depthHeight-64,depthWidth,'uint16');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
colorSkin = uint8(zeros(colorHeightScaled,colorWidthScaled,3));

meanGMM = zeros(dim,G,numbColorPixel);
covGMM = ones(G,numbColorPixel)*20;
wGMM = zeros(G,numbColorPixel);
wCovGMM = zeros(G,numbColorPixel);
orderGMM = repmat((1:G)',1,numbColorPixel); 

meanGMM(:,1,:) = reshape(permute(meanBG, [3,1,2]), dim,1,[]);
covGMM(1,:) = reshape(covBG, 1, []);
wGMM(1,:) = 1;
wCovGMM = bsxfun(@rdivide, wGMM, covGMM); % descend order

% depth stream figure
figure, hDepth = imshow(depthCali,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar

% color stream figure
figure, hColor = imshow(colorCaliScale,[]);
title('Color Source');

% skin stream figure
figure, hSkin = imshow(colorSkin,[]);
title('Skin Source');

k = 1;
while k <= 200000
    tic;
    % Get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    skinFGMask = zeros(colorHeightScaled,colorWidthScaled);
    
    % Before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % Copy data to Matlab matrices        
        depth = k2.getDepth;
        color = k2.getColor;
        depthCali = depth(27:386, :);
        colorCali = color(:, 221:1756, :);         
        
        % update depth figure
        depthCali(depthCali>outOfRange) = outOfRange; % truncate depht
        set(hDepth,'CData',depthCali); 

        % update color figure
        colorCaliScale = imresize(colorCali,colorScale);
        set(hColor,'CData',colorCaliScale); 

    end
    
    depthFGPixel = abs(bsxfun(@minus, double(depthCali), depthMean));
    depthFGIndex = find(depthFGPixel>depthStandard3);
    [depthFGRow,depthFGCol] = ind2sub([colorHeightScaled,colorWidthScaled],depthFGIndex); 
    depthFGCol = depthFGCol + round(bsxfun(@plus,bsxfun(@rdivide, 18015.57,double(depthCali(depthFGIndex))), -22.072174));
    depthFG = [depthFGRow, depthFGCol]';
    
    colorFG = dilatation(depthFG,structuralElement,colorHeightScaled,colorWidthScaled);
    colorResh = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
    colorFGIndex = sub2ind([colorHeightScaled,colorWidthScaled], colorFG(1,:),colorFG(2,:));
    colorFGRGB = double(colorResh(:, colorFGIndex));
    numbColorFG = length(colorFGIndex);
    
    meanGMM4ColorFG = meanGMM(:,:,colorFGIndex);
    covGMM4ColorFG = covGMM(:,colorFGIndex);
    wGMM4ColorFG = wGMM(:,colorFGIndex);
    wCovGMM4ColorFG = wCovGMM(:,colorFGIndex);
    orderGMM4ColorFG = orderGMM(:,colorFGIndex);
    [minBias, minBiasIndex] = min( sum(abs(bsxfun(@minus,colorFGRGB,permute(meanGMM4ColorFG,[1,3,2]))),1),[],3);
    covGMMIndex = bsxfun(@plus, (0:(numbColorFG-1))*G, minBiasIndex);
    ifMatch = bsxfun(@lt, minBias, covGMM4ColorFG(covGMMIndex));%
    numbColorFGTmp = 1:numbColorFG;
    ifMatchIndex = numbColorFGTmp(ifMatch);
    ifNotMatchIndex = numbColorFGTmp(~ifMatch);
    BGIndexBGSubtr = false(1,numbColorFG);
    
    if ~isempty(ifNotMatchIndex)
        replaceIndex = bsxfun(@plus, (ifNotMatchIndex-1)*G, orderGMM4ColorFG(end,ifNotMatchIndex));
        meanGMM4ColorFG(:,replaceIndex) = colorFGRGB(:,ifNotMatchIndex);
        covGMM4ColorFG(replaceIndex) = covInitHigh;
        wGMM4ColorFG(:,ifNotMatchIndex) = bsxfun(@times, wGMM4ColorFG(:,ifNotMatchIndex),...
            (1 + bsxfun(@rdivide,(wGMM4ColorFG(replaceIndex)-wInitLow),(1-wGMM4ColorFG(replaceIndex)) ) ) );
        wGMM4ColorFG(replaceIndex) = wInitLow;
        wCovGMM4ColorFG(end,ifNotMatchIndex) = bsxfun(@rdivide, wInitLow, covInitHigh);
    end
    if ~isempty(ifMatchIndex)
        updateIndex = bsxfun(@plus,(ifMatchIndex-1)*G, minBiasIndex(ifMatchIndex));
        %wUpdateMulti = bsxfun(@plus, bsxfun(@times, ifMatch, (1-alpha)), bsxfun(@minus, 1, ifMatch));
        %wGMM4ColorFG = bsxfun(@times,wGMM4ColorFG,wUpdateMulti');
        wGMM4ColorFG(:,ifMatchIndex) = wGMM4ColorFG(:,ifMatchIndex)*(1-alpha);
        wGMM4ColorFG(updateIndex) = wGMM4ColorFG(updateIndex) + alpha;
        colorFGUpdate = colorFGRGB(:,ifMatchIndex);
        meanGMM4ColorFGUpdate = meanGMM4ColorFG(:,updateIndex);
        covGMM4ColorFGUpdate =covGMM4ColorFG(updateIndex);
        gaussianTmp = exp(-1/2*bsxfun(@rdivide, sum(bsxfun(@power, ...
            bsxfun(@minus, colorFGUpdate, meanGMM4ColorFGUpdate), 2),1), covGMM4ColorFGUpdate)); 
        gaussian4ColorFGUpdate = bsxfun(@times, (2*pi)^(-dim/2)*bsxfun(@power, covGMM4ColorFGUpdate, -dim/2), gaussianTmp);
        rho = gaussian4ColorFGUpdate*alpha;
        one_rho = 1-rho;
        meanGMM4ColorFG(:,updateIndex) = bsxfun(@plus, bsxfun(@times, one_rho, meanGMM4ColorFG(:,updateIndex)),...
            bsxfun(@times, rho, colorFGUpdate));
        covGMM4ColorFG(updateIndex) = bsxfun(@plus, bsxfun(@times, one_rho, covGMM4ColorFG(updateIndex)),...
            bsxfun(@times, rho, sum(bsxfun(@power, bsxfun(@minus, colorFGUpdate, meanGMM4ColorFG(:,updateIndex)), 2), 1)));
        wCovGMM4ColorFG(updateIndex) = bsxfun(@rdivide,wGMM4ColorFG(updateIndex),covGMM4ColorFG(updateIndex));
        [~,orderTmp] = sort(wCovGMM4ColorFG(:,ifMatchIndex),'descend'); %
        orderGMM4ColorFG(:,ifMatchIndex) = orderTmp;
        orderGMM(:,colorFGIndex) = orderGMM4ColorFG;
        for i = 1:length(ifMatchIndex)
            ifMatchIndexTmp = ifMatchIndex(i);
            FGGMM = find(orderGMM4ColorFG(:,ifMatchIndexTmp)==minBiasIndex(ifMatchIndexTmp));%
            sumTmp = sum(wGMM4ColorFG(orderGMM4ColorFG(1:FGGMM,ifMatchIndexTmp),ifMatchIndexTmp));
            BGIndexBGSubtr(ifMatchIndexTmp) = (sumTmp<BGPerc)||((sumTmp > BGPerc)&&(sumTmp-wGMM4ColorFG(minBiasIndex(ifMatchIndexTmp),ifMatchIndexTmp) < BGPerc));
        end
    end
    meanGMM(:,:,colorFGIndex) = meanGMM4ColorFG;
    covGMM(:,colorFGIndex) = covGMM4ColorFG;
    wGMM(:,colorFGIndex) = wGMM4ColorFG;
    wCovGMM(:,colorFGIndex) = wCovGMM4ColorFG;
    colorFGIndex(BGIndexBGSubtr) = [];
    colorFGRGB(:,BGIndexBGSubtr) = [];
    colorSkinTmp = zeros(size(colorSkinTmp));
    if ~isempty(colorFGIndex)
        colorFGLDA = bestDirHandLDA'*colorFGRGB;
        colorFGLDABias = bsxfun(@minus, colorFGLDA, meanHandLDA);
        if (modeHandLDA == 2)
            ellipse1 = sum(bsxfun(@rdivide, bsxfun(@power,colorFGLDABias,2), diag(covHandLDAWeighted)));
            ellipse2 = bsxfun(@times, bsxfun(@times, colorFGLDABias(1,:),colorFGLDABias(2,:)), tworho__sigmaProduct);
            ellipse = bsxfun(@minus, ellipse1, ellipse2);
            ifHandColor = bsxfun(@lt,ellipse,one_rho2);
        else
            ifHandColor = bsxfun(@lt, colorFGLDABias, sqrt(covHandLDA));
        end
        colorSkinTmp(:,colorFGIndex(ifHandColor)) = colorFGRGB(:,ifHandColor);
        %colorSkinTmp(:,colorFGIndex) = colorFGRGB;
    end
    colorSkin = uint8(reshape(colorSkinTmp',colorHeightScaled,colorWidthScaled,3));
    set(hSkin,'CData',colorSkin);

%     depthFGRow(depthFGRow>colorHeightScaled) = colorHeightScaled;
%     depthFGCol(depthFGCol>colorWidthScaled) = colorWidthScaled;
%     depthFGRow(depthFGRow<1) = 1;
%     depthFGCol(depthFGCol<1) = 1;
%     colorFGIndex = sub2ind([colorHeightScaled,colorWidthScaled], depthFGRow,depthFGCol);
%     skinFGMask(colorFGIndex) = 1;
%     colorSkin = uint8(bsxfun(@times, double(colorCaliScale), skinFGMask));
%     set(hSkin,'CData',colorSkin);
    
    k = k + 1;
    pause(0.02)
    toc
end

% Close kinect object
k2.delete;

close all;



%% without Depth
close all

startPixel = [91,129];
endPixel = [270,384];

G = 4; % number of the gaussian function in each pixel 
alpha = 0.2;
covInitHigh = 20;
BGPerc = 0.6;

% Create Kinect 2 object and initialize it
% Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
k2 = Kin2('color');

% Create matrices for the images
color = zeros(colorHeight,colorWidth,3,'uint8');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
colorSkin = uint8(zeros(colorHeightScaled,colorWidthScaled,3));

[colorS1,colorS2] = ind2sub([180 256],1:180*256);
colorFG = [colorS1+90;colorS2+128];
colorFGIndex = sub2ind([colorHeightScaled,colorWidthScaled], colorFG(1,:),colorFG(2,:));
numbColorFG = length(colorFGIndex);

meanGMM4ColorFG = zeros(dim,G,numbColorFG);
covGMM4ColorFG = ones(G,numbColorFG)*20;
wGMM4ColorFG = zeros(G,numbColorFG);
wCovGMM4ColorFG = zeros(G,numbColorFG);
orderGMM4ColorFG = repmat((1:G)',1,numbColorFG);

meanBGTmp = reshape(permute(meanBG, [3,1,2]), dim,1,[]); 
meanGMM4ColorFG(:,1,:) = meanBGTmp(:,:,colorFGIndex);
covBGTmp = reshape(covBG, 1, []);
covGMM4ColorFG(1,:) = covBGTmp(:,colorFGIndex);
wGMM4ColorFG(1,:) = 1;
wCovGMM = bsxfun(@rdivide, wGMM4ColorFG, covGMM4ColorFG); % descend order

% color stream figure
figure, hColor = imshow(colorCaliScale,[]);
title('Color Source');

% skin stream figure
figure, hSkin = imshow(colorSkin,[]);
title('Skin Source');

k = 1;
while k <= 200000
    tic;
    % Get frames from Kinect and save them on underlying buffer
    validData = k2.updateData;
    skinFGMask = zeros(colorHeightScaled,colorWidthScaled);
    
    % Before processing the data, we need to make sure that a valid
    % frame was acquired.
    if validData
        % Copy data to Matlab matrices        
        color = k2.getColor;
        colorCali = color(:, 221:1756, :);         

        % update color figure
        colorCaliScale = imresize(colorCali,colorScale);
        set(hColor,'CData',colorCaliScale); 

    end
    
    colorResh = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
    colorFGRGB = double(colorResh(:, colorFGIndex));
    
    [minBias, minBiasIndex] = min( sum(abs(bsxfun(@minus,colorFGRGB,permute(meanGMM4ColorFG,[1,3,2]))),1),[],3);
    covGMMIndex = bsxfun(@plus, (0:(numbColorFG-1))*G, minBiasIndex);
    ifMatch = bsxfun(@lt, minBias, covGMM4ColorFG(covGMMIndex));%
    numbColorFGTmp = 1:numbColorFG;
    ifMatchIndex = numbColorFGTmp(ifMatch);
    ifNotMatchIndex = numbColorFGTmp(~ifMatch);
    FGIndexBGSubtr = true(1,numbColorFG);
    if ~isempty(ifNotMatchIndex)
        replaceIndex = bsxfun(@plus, (ifNotMatchIndex-1)*G, orderGMM4ColorFG(end,ifNotMatchIndex));
        meanGMM4ColorFG(:,replaceIndex) = colorFGRGB(:,ifNotMatchIndex);
        covGMM4ColorFG(replaceIndex) = covInitHigh;
        wGMM4ColorFG(:,ifNotMatchIndex) = bsxfun(@times, wGMM4ColorFG(:,ifNotMatchIndex),...
            (1+bsxfun(@rdivide,(wGMM4ColorFG(replaceIndex)-wInitLow),(1-wGMM4ColorFG(replaceIndex)))));
        wGMM4ColorFG(replaceIndex) = wInitLow;
        wCovGMM4ColorFG(end,ifNotMatchIndex) = bsxfun(@rdivide, wInitLow, covInitHigh);
    end
    if ~isempty(ifMatchIndex)
        updateIndex = bsxfun(@plus,(ifMatchIndex-1)*G, minBiasIndex(ifMatchIndex));
        wGMM4ColorFG(:,ifMatchIndex) = wGMM4ColorFG(:,ifMatchIndex)*(1-alpha);
        wGMM4ColorFG(updateIndex) = wGMM4ColorFG(updateIndex) + alpha;
        colorFGUpdate = colorFGRGB(:,ifMatchIndex);
        meanGMM4ColorFGUpdate = meanGMM4ColorFG(:,updateIndex);
        covGMM4ColorFGUpdate =covGMM4ColorFG(updateIndex);
        gaussianTmp = exp(-1/2*bsxfun(@rdivide, sum(bsxfun(@power, bsxfun(@minus, colorFGUpdate, meanGMM4ColorFGUpdate), 2),1), covGMM4ColorFGUpdate)); 
        gaussian4ColorFGUpdate = bsxfun(@times, (2*pi)^(-dim/2)*bsxfun(@power, covGMM4ColorFGUpdate, -dim/2), gaussianTmp);
        rho = gaussian4ColorFGUpdate*alpha;
        one_rho = 1-rho;
        meanGMM4ColorFG(:,updateIndex) = bsxfun(@plus, bsxfun(@times, one_rho, meanGMM4ColorFG(:,updateIndex)),...
            bsxfun(@times, rho, colorFGUpdate));
        covGMM4ColorFG(updateIndex) = bsxfun(@plus, bsxfun(@times, one_rho, covGMM4ColorFG(updateIndex)),...
            bsxfun(@times, rho, sum(bsxfun(@power, bsxfun(@minus, colorFGUpdate, meanGMM4ColorFG(:,updateIndex)), 2), 1)));
        wCovGMM4ColorFG(updateIndex) = bsxfun(@rdivide,wGMM4ColorFG(updateIndex),covGMM4ColorFG(updateIndex));
        [~,orderTmp] = sort(wCovGMM4ColorFG(:,ifMatchIndex),'descend'); %
        orderGMM4ColorFG(:,ifMatchIndex) = orderTmp;
        for i = 1:length(ifMatchIndex)
            ifMatchIndexTmp = ifMatchIndex(i);
            FGGMM = find(orderGMM4ColorFG(:,ifMatchIndexTmp)==minBiasIndex(ifMatchIndexTmp));%
            sumTmp = sum(wGMM4ColorFG(orderGMM4ColorFG(1:FGGMM,ifMatchIndexTmp),ifMatchIndexTmp));
            FGIndexBGSubtr(ifMatchIndexTmp) = ~((sumTmp<BGPerc)||((sumTmp > BGPerc)&&(sumTmp-wGMM4ColorFG(minBiasIndex(ifMatchIndexTmp),ifMatchIndexTmp) < BGPerc)));
        end
    end
    colorSkinTmp = zeros(size(colorSkinTmp));
    colorFGRGBFG = colorFGRGB(:,FGIndexBGSubtr);
    if ~isempty(colorFGRGBFG)
        colorFGLDA = bestDirHandLDA'*colorFGRGBFG;
        colorFGLDABias = bsxfun(@minus, colorFGLDA, meanHandLDA);
        if (modeHandLDA == 2)
            ellipse1 = sum(bsxfun(@rdivide, bsxfun(@power,colorFGLDABias,2), diag(covHandLDAWeighted)));
            ellipse2 = bsxfun(@times, bsxfun(@times, colorFGLDABias(1,:),colorFGLDABias(2,:)), tworho__sigmaProduct);
            ellipse = bsxfun(@minus, ellipse1, ellipse2);
            ifHandColor = bsxfun(@lt,ellipse,one_rho2);
        else
            ifHandColor = bsxfun(@lt, colorFGLDABias, covHandLDA);
        end
        colorSkinTmp(:,colorFGIndex(FGIndexBGSubtr(ifHandColor))) = colorFGRGB(:,FGIndexBGSubtr(ifHandColor));
    end
    colorSkin = uint8(reshape(colorSkinTmp',colorHeightScaled,colorWidthScaled,3));
    set(hSkin,'CData',colorSkin);
    k = k + 1;
    
    pause(0.02)
    toc
end

% Close kinect object
k2.delete;

close all;
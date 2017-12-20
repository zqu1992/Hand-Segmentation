addpath('Mex');
clear all

pause(5)

pSkin = skinColorModel(2,100);

pause(3)
%%

[depthMean, depthCov] = depthBackgroundInit(200);
depthStandard = sqrt(depthCov);
depthStandard3 = depthStandard * 3;
depthStandard6 = depthStandard * 6;

%%

close all

% Create Kinect 2 object and initialize it
% Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
% 'face' and 'HDface'
k2 = Kin2('color','depth','infrared');

% [s1,s2] = ind2sub([11 11],1:121);
% structuralElement = [s1-6;s2-6];

%[s1,s2] = ind2sub([5 5],1:25);
%structuralElement = [s1-3;s2-3];

structuralElement = [0;0];


% images sizes
depthWidth = 512; depthHeight = 424; outOfRange = 4000;
colorWidth = 1920; colorHeight = 1080;

% Color image is to big, let's scale it down
colorScale = 1/3;

% Create matrices for the images
depth = zeros(depthHeight,depthWidth,'uint16');
%infrared = zeros(depthHeight,depthWidth,'uint16');
color = zeros(colorHeight,colorWidth,3,'uint8');

depthCali = zeros(depthHeight-64,depthWidth,'uint16');
%infraredCali = zeros(depthHeight-64,depthWidth,'uint16');
colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
colorHeightScaled = colorHeight/3;
colorWidthScaled = (colorWidth-384)/3;
numbColorPixel = colorHeightScaled*colorWidthScaled;
colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
colorSkin = zeros(colorHeightScaled,colorWidthScaled,3);


% depth stream figure
figure, hDepth = imshow(depthCali,[0 outOfRange]);
title('Depth Source')
colormap('Jet')
colorbar


% color stream figure
figure, hColor = imshow(colorCaliScale,[]);
title('Color Source');


% infrared stream figure
%figure, hInfrared = imshow(infraredCali);
%title('Infrared Source');

% skin stream figure
figure, hSkin = imshow(colorSkin,[]);
title('Skin Source');

%%
% Transfer matrix from RGB to YCbCr
TranMat = [0.299 0.587 0.114; -0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];
TranMatChro = [-0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];

k = 0;

while k < 200000
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
        %infrared = k2.getInfrared;
        depthCali = depth(27:386, :);
        %infraredCali = infrared(27:386, :);
        colorCali = color(:, 221:1756, :);
             
        
        % update depth figure
        depthCali(depthCali>outOfRange) = outOfRange; % truncate depht
        set(hDepth,'CData',depthCali); 

        % update color figure
        colorCaliScale = imresize(colorCali,colorScale);
        set(hColor,'CData',colorCaliScale); 

        % update infrared figure
        %infrared = imadjust(infrared,[0 0.2],[0.5 1]);
        %infraredCali = imadjust(infraredCali,[],[],0.5);
        %set(hInfrared,'CData',infraredCali); 

    end
    
    depthFGPixel = abs(bsxfun(@minus, double(depthCali), depthMean));
    depthFGIndex = find(depthFGPixel>depthStandard6);
    [depthFGRow,depthFGCol] = ind2sub([colorHeightScaled,colorWidthScaled],depthFGIndex); 
    depthFGCol = depthFGCol + round(bsxfun(@plus,bsxfun(@rdivide, 18015.57,double(depthCali(depthFGIndex))), -22.072174));
    depthFG = [depthFGRow, depthFGCol]';
    
    colorFG = dilatation(depthFG,structuralElement,colorHeightScaled,colorWidthScaled);
    colorResh = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
    colorFGIndex = sub2ind([colorHeightScaled,colorWidthScaled], colorFG(1,:),colorFG(2,:));
    colorFGRGB = double(colorResh(:, colorFGIndex));
    colorFGChro = zeros(2,length(colorFGRGB));
    colorFGChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', colorFGRGB)) + 128;
    colorFGChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', colorFGRGB)) + 128;
    colorFGChro = round(colorFGChro);
    colorFGChro(colorFGChro<0) = 0;
    colorFGChro(colorFGChro>255) = 255;
    colorFGChroIndex = sub2ind(size(pSkin), colorFGChro(1,:), colorFGChro(2,:)); 
    pSkinTmp = pSkin(colorFGChroIndex);
    skinFGMask(colorFGIndex(pSkinTmp>1E-10)) = 1; %%1E-10
    colorSkin = uint8(bsxfun(@times, double(colorCaliScale), skinFGMask));
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
  
    pause(0.02);
    toc
end

% Close kinect object
k2.delete;

close all;
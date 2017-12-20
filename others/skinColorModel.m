function [ pSkin, pNonSkin ] = skinColorModel( KSkin, KNonSkin, tSkin, tNonSkin )
%   This function is used to collect training data for hand color and
%   environmental color around workbench, and return the look up table of
%   probability distribution in CbCr color space (256*256) of skin and 
%   non-skin. The model of pdf is gaussian mixture model.  
%   

    %%%------ Parameter Setting ------%%%
    addpath('Mex');
    close all;
    % chrominance component
    CbCrSkin = zeros(256,256);
    CbCrNonSkin = zeros(256,256);
    % images sizes
    colorWidth = 1920; colorHeight = 1080;
    % Color image is to big, let's scale it down
    colorScale = 1/3;                                                      
    colorHeightScaled = colorHeight/3;
    colorWidthScaled = (colorWidth-384)/3;

    % create matrices for the images
    color = zeros(colorHeight,colorWidth,3,'uint8');
    colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
    colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
    colorOne = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
    colorRealTime = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
    % transfer matrix from RGB to YCbCr
    TranMat = [0.299 0.587 0.114; -0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];
    TranMatChro = [-0.1687 -0.3313 0.5; 0.5 -0.4187 -0.0813];

    
    %%%------ Skin Color Training Data Collection ------%%%
    disp('Please set the Kinect in front of a black cloth near the workbench.');
    disp('If finished, please press any key to continue...')
    pause;
    disp('Please put your hands on the black cloth, and rotate them slowly.');
    disp('The collection will be taken after 3 seconds.')
    pause(3);
    % create Kinect 2 object and initialize it
    % available sources: 'color', 'depth', 'infrared', 'body_index', 
    % 'body', 'face' and 'HDface'
    k2 = Kin2('color');                                                     
    % color stream figure
    figure, hSkin = imshow(colorCaliScale,[]);
    title('Skin Color');
    % loop last few seconds
    i = 0;
    while i < tSkin
        % get frames from Kinect and save them on underlying buffer
        validData = k2.updateData;
        % before processing the data, we need to make sure that a valid
        % frame was acquired.
        if validData
            % copy data to Matlab matrices
            color = k2.getColor;
            colorCali = color(:, 221:1756, :);
            % update color figure
            colorCaliScale = imresize(colorCali,colorScale);
            set(hSkin,'CData',colorCaliScale);
        end
        colorSumRGB = sum(colorCaliScale, 3);
        skinIndex = find(colorSumRGB > 600);
        if ~isempty(skinIndex)
            colorResh = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
            skinRGB = double(colorResh(:, skinIndex));
            skinChro = zeros(2,length(skinRGB));
            skinChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', skinRGB)...
                ,1) + 128;
            skinChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', skinRGB)...
                ,1) + 128;
            skinChro = round(skinChro);
            skinChro(skinChro<0) = 0;
            skinChro(skinChro>255) = 255;
            statisSkinChro = sub2ind(size(CbCrSkin),skinChro(1,:)+1,...
                skinChro(2,:)+1);
            tbl = tabulate(statisSkinChro);
            CbCrSkin(tbl(:,1)) = CbCrSkin(tbl(:,1)) + tbl(:,2);
        end
        i = i + 1;
        pause(0.02)
    end
    % close kinect object
    k2.delete;
    close all;
    
    
    %%%------ environmental color training data collection ------%%%
    % create Kinect 2 object and initialize it
    % available sources: 'color', 'depth', 'infrared', 'body_index', 
    % 'body', 'face' and 'HDface'
    k2 = Kin2('color');
    % color stream figure
    figure, hNonSkin = imshow(colorCaliScale,[]);
    title('Non Skin Color');
    % Loop last 5s
    i = 0;
    while i < tNonSkin
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
            set(hNonSkin,'CData',colorCaliScale);
        end
        nonSkinRGB = double(reshape(permute(colorCaliScale, [3,1,2]), 3, []));
        nonSkinChro = zeros(2,length(nonSkinRGB));
        nonSkinChro(1,:) = sum(bsxfun(@times, TranMatChro(1,:)', ...
            nonSkinRGB),1) + 128;
        nonSkinChro(2,:) = sum(bsxfun(@times, TranMatChro(2,:)', ...
            nonSkinRGB),1) + 128;
        nonSkinChro = round(nonSkinChro);
        nonSkinChro(nonSkinChro<0) = 0;
        nonSkinChro(nonSkinChro>255) = 255;
        statisNonSkinChro = sub2ind(size(CbCrNonSkin),nonSkinChro(1,:)+1,...
            nonSkinChro(2,:)+1);
        tbl = tabulate(statisNonSkinChro);
        CbCrNonSkin(tbl(:,1)) = CbCrNonSkin(tbl(:,1)) + tbl(:,2);
        i = i + 1;
        pause(0.02)
    end
    % Close kinect object
    k2.delete;
    close all;
    
    %%%------ GMM PDF (CbCr) of skin and non-skin calculation ------%%%
    [piSkin, meanSkin, covSkin] = emGmmKmeans(CbCrSkin,KSkin);
    pSkin = zeros(256,256);
    gaussian = zeros(256*256,KSkin);
    for i = 1:size(gaussian,1)
        for j = 1:KSkin
            [r,c] = ind2sub(size(pSkin),i);
            gaussian(i,j) = mvnpdf([r;c], meanSkin(:,j),covSkin(:,:,j));
        end
    end
    pSkinK = bsxfun(@times, piSkin, gaussian);
    pSum = sum(pSkinK,2);
    pSkin(:) = pSum(:);

    [piNonSkin, meanNonSkin, covNonSkin] = emGmmKmeans(CbCrNonSkin,KNonSkin);
    pNonSkin = zeros(256,256);
    gaussian = zeros(256*256,KNonSkin);
    for i = 1:size(gaussian,1)
        for j = 1:KNonSkin
            [r,c] = ind2sub(size(pNonSkin),i);
            gaussian(i,j) = mvnpdf([r;c], meanNonSkin(:,j),covNonSkin(:,:,j));
        end
    end
    pNonSkinK = bsxfun(@times, piNonSkin, gaussian);
    pSum = sum(pNonSkinK,2);
    pNonSkin(:) = pSum(:);
end


function [meanBG, covBG] = AdapBGSubtInit( tBGInit )
%   This function is used for initialization of adaptive background
%   subtraction.
%   Input: the number of frames used for initialization
%   Output: the mean vector of background RGB value, the covariance matrix 
%   of background RGB value
    
    % add mex file into path
    addpath('Mex');
    % close all windows
    close all
    % set the dimension of used color space (RGB)
    dim = 3;
    % create Kinect 2 object and initialize it
    % available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
    % 'face' and 'HDface'
    k2 = Kin2('color');
    % set the image sizes
    colorWidth = 1920; colorHeight = 1080;
    % scale the color image down
    % scale factor
    colorScale = 1/3;
    % the color image size after scaling
    colorHeightScaled = colorHeight/3;
    colorWidthScaled = (colorWidth-384)/3;
    % create matrices of the original source and image source after
    % calibration
    color = zeros(colorHeight,colorWidth,3,'uint8');
    colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
    colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');
    % define the
    meanBG = zeros(colorHeightScaled,colorWidthScaled,3);
    covBG = zeros(colorHeightScaled,colorWidthScaled);
    % color stream image after scaling
    figure, hBGInit = imshow(colorCaliScale,[]);
    title('Background subtrc Color Source');

    % initialization loop
    i = 1;
    while i <= tBGInit
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
            set(hBGInit,'CData',colorCaliScale); 
        end
        % data type conversion
        colorBG = double(colorCaliScale);
        % calculate the mean vector of background RGB value
        meanBG = bsxfun(@plus, bsxfun(@times, meanBG, (i-1)/i), bsxfun(@rdivide, colorBG, i));
        % calculate the covariance matrix of background RGB value
        covBGTmp = bsxfun(@rdivide, sum(bsxfun(@power, bsxfun(@minus, colorBG, meanBG), 2),3), dim);
        covBG = bsxfun(@plus, bsxfun(@times, covBG, (i-1)/i), bsxfun(@rdivide, covBGTmp, i));
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
end


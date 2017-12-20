% Create on Dec, 2016
% @author: zhongnan qu
function [ depthMean, depthCov ] = depthBackgroundInit( tDepth )
%   This function is used for the collection of stationary background 
%   depth value, which is the initialization of depth background 
%   subtraction.
%   Input: the number of frames used for initialization
%   Output: the mean of background depth value, the variance of background depth
%   value

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

    % create matrices of the original source and image source after calibration
    depth = zeros(depthHeight,depthWidth,'uint16');
    infrared = zeros(depthHeight,depthWidth,'uint16');
    depthCali = zeros(depthHeight-64,depthWidth,'uint16');
    infraredCali = zeros(depthHeight-64,depthWidth,'uint16');
    % create a matrix to gather the background depth data
    depthSet = zeros(depthHeight-64, depthWidth, tDepth);
    % create a matrix of out of range depth value
    depthOutOfRange = zeros(depthHeight-64, depthWidth);

    % depth stream image
    figure, hDepthBGInit = imshow(depthCali,[0 outOfRange]);
    title('Depth Source')
    colormap('Jet')
    colorbar
    % infrared stream image
    figure, hInfraredBGInit = imshow(infraredCali);
    title('Infrared Source');

    % initialization loop
    i = 1;
    while i <= tDepth
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
            depthCali(depthCali>outOfRange) = outOfRange; % truncate depht
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

end


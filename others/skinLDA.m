function [ bestDir, meanHandLDA, covHandLDA, weightHandLDA] = skinLDA( tHand, tSpace, mode)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    
    addpath('Mex');
    close all
    
    dim = 3;
    maxJLDA = -1e10;
    maxJLDAOrth = -1e10;
    %
    dirVec = [0 0 1]';
    delta = 2*pi*1/200;
    for i = 1:100
        J = ceil(2*pi*sin(pi/100*i)/4/delta)*4;
        for j = 0:J-1
            [a, b, c] = sph2cart(j*2*pi/J,pi/2-pi/100*i,1);
            dirVec(:,end+1) = [a;b;c];
        end
    end
    if (mode == 1) 
        bestDir = zeros(3,1);
        bestDirLast = zeros(3,1);
    else 
        bestDir = zeros(3,2);
        bestDirLast = zeros(3,2);
    end
    
    % Create Kinect 2 object and initialize it
    % Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
    % 'face' and 'HDface'
    k2 = Kin2('color');

    % images sizes
    colorWidth = 1920; colorHeight = 1080;

    % Color image is to big, let's scale it down
    colorScale = 1/3;

    % Create matrices for the images
    color = zeros(colorHeight,colorWidth,3,'uint8');
    colorCali = zeros(colorHeight,colorWidth-384,3,'uint8');
    colorHeightScaled = colorHeight/3;
    colorWidthScaled = (colorWidth-384)/3;
    numbColorPixel = colorHeightScaled*colorWidthScaled;
    colorCaliScale = zeros(colorHeightScaled,colorWidthScaled,3,'uint8');

    %%%%% 2 principal component
    tbHandRGB = zeros(256*256*256,1);

    % Transfer matrix from RGB to LDA classification
    
    % color stream figure
    figure, hHandColor = imshow(colorCaliScale,[]);
    title('Hand Color Source');


    % Loop last 5s
    i = 0;
    while i < tHand
        tic;
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
            set(hHandColor,'CData',colorCaliScale); 

        end

        colorSumRGB = sum(colorCaliScale, 3);
        handIndex = find(colorSumRGB > 600);
        if ~isempty(handIndex)
            colorResh = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
            covHandRGB = double(colorResh(:, handIndex));
            statisHandRGB = sub2ind([256,256,256],covHandRGB(1,:)+1,covHandRGB(2,:)+1,covHandRGB(3,:)+1);
            tbHand = tabulate(statisHandRGB);
            tbHandRGB(tbHand(:,1)) = tbHandRGB(tbHand(:,1)) + tbHand(:,2);
        end
        i = i + 1;

        toc
        pause(0.02)

        
    end
    % Close kinect object
    k2.delete;
    close all;
    sum(tbHandRGB)
    handRGBIndex = find(tbHandRGB>0);
    handRGBNumb = tbHandRGB(handRGBIndex);
    [ind1,ind2,ind3] = ind2sub([256,256,256],handRGBIndex);
    dataHandRGB = [ind1,ind2,ind3];
    meanHandRGB = (sum(bsxfun(@times,dataHandRGB,handRGBNumb))/sum(handRGBNumb))';
    handRGBTmp = bsxfun(@minus, dataHandRGB', meanHandRGB); 
    handRGBTmp1 = bsxfun(@times,reshape(handRGBTmp,3,[],length(handRGBIndex)),reshape(handRGBTmp,[],3,length(handRGBIndex)));
    covHandRGB = sum(bsxfun(@times, handRGBTmp1, reshape(handRGBNumb,1,[],length(handRGBIndex))),3)/sum(handRGBNumb);
    
    pause(5);
    
    close all;
    % Create Kinect 2 object and initialize it
    % Available sources: 'color', 'depth', 'infrared', 'body_index', 'body',
    % 'face' and 'HDface'
    k2 = Kin2('color');
    %
    tbSpaceRGB = zeros(256*256*256,1);
    % color stream figure
    figure, hSpaceColor = imshow(colorCaliScale,[]);
    title('Space Color Source');
    % Loop last 5s
    i = 0;

    while i < tSpace
        tic
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
            set(hSpaceColor,'CData',colorCaliScale); 
        end
        spaceRGB = reshape(permute(colorCaliScale, [3,1,2]), 3, []);
        statisSpaceRGB = sub2ind([256,256,256],spaceRGB(1,:)+1,spaceRGB(2,:)+1,spaceRGB(3,:)+1);
        tbSpace = tabulate(statisSpaceRGB);
        tbSpaceRGB(tbSpace(:,1)) = tbSpaceRGB(tbSpace(:,1)) + tbSpace(:,2);
        i = i + 1;
        toc
        pause(0.02)
    end
    % Close kinect object
    k2.delete;
    close all;
    sum(tbSpaceRGB)
    spaceRGBIndex = find(tbSpaceRGB>0);
    spaceRGBNumb = tbSpaceRGB(spaceRGBIndex);
    [ind1,ind2,ind3] = ind2sub([256,256,256],spaceRGBIndex);
    spaceRGBData = [ind1,ind2,ind3]';
    
    for K = 6:8
        bestDirLast = bestDir;
        JKmeansLast = 2E8;
        JKmeans = 1E8;
        [~,index] = sort(spaceRGBNumb,'descend');
        index = index(1:K);
        % total number of observation 
        N = sum(spaceRGBNumb);
        numb = length(spaceRGBNumb);
        meanK = spaceRGBData(:,index);
        covK = zeros(dim,dim,K);
        sumObser = zeros(K,numb);
        while abs(JKmeansLast - JKmeans) >= 1E-8
            euc_dist_quad_k = reshape(sum(bsxfun(@minus, reshape(spaceRGBData,dim,numb,[]), reshape(meanK,dim,[],K)).^2), numb, K);
            [euc_dist_quad,w] = min(euc_dist_quad_k,[],2);
            JKmeansLast = JKmeans;
            k = 1:K;
            sumObser = bsxfun(@times, bsxfun(@eq,w,k), spaceRGBNumb)'; % each pixel in which cluster and has how many observations, K*numb
            meanKtmp = sum(bsxfun(@times, spaceRGBData', reshape(sumObser',numb,[],K)));
            meanK = bsxfun(@rdivide, reshape(meanKtmp, dim, K), sum(sumObser,2)');
            zeroMean = bsxfun(@minus, reshape(spaceRGBData,dim,numb,[]), reshape(meanK,dim,[],K));
            zeroMeanTmp = bsxfun(@times, zeroMean, reshape(sumObser',[],numb,K));
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
            JKmeansTmp = sum(covK,3);
            JKmeans = trace(JKmeansTmp);
        end
        numbK = sum(sumObser,2);
        
        Sw = bsxfun(@plus, covK, covHandRGB);
        meanDiff = bsxfun(@minus, meanK, meanHandRGB);
        Sb = bsxfun(@times, reshape(meanDiff,dim,[],K), reshape(meanDiff,[],dim,K)); 
        maxJLDA = -1e10;
        maxJLDAOrth = -1e10;
        for i = 1:length(dirVec)
            SwTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVec(:,i),Sw),1), dirVec(:,i)'),2);
            SbTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVec(:,i),Sb),1), dirVec(:,i)'),2);
            JLDA = sum(bsxfun(@times, bsxfun(@rdivide, SbTmp, SwTmp), reshape(numbK,1,[],K)),3); 
            if (JLDA>maxJLDA)
                maxJLDA = JLDA;
                bestDir(:,1) = dirVec(:,i);
            end
        end
        if (mode == 2)
            eTheta = zeros(3,1);
            ePhi = zeros(3,1);
            sqrtTmp = sqrt(bestDir(1,1)^2+bestDir(2,1)^2);
            eTheta(1) = bestDir(3,1)*bestDir(1,1)/sqrtTmp;
            eTheta(2) = bestDir(3,1)*bestDir(2,1)/sqrtTmp;
            eTheta(3) = -sqrtTmp;
            ePhi(1) = -bestDir(2,1)/sqrtTmp;
            ePhi(2) = bestDir(1,1)/sqrtTmp;
            ePhi(3) = 0;
            
            for i = 1:360:2*pi
                dirVecOrth = eTheta*sin(i)+ePhi*cos(i);
                SwOrthTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVecOrth,Sw),1), dirVecOrth'),2);
                SbOrthTmp = sum(bsxfun(@times, sum(bsxfun(@times, dirVecOrth,Sb),1), dirVecOrth'),2);
                JLDAOrth = sum(bsxfun(@times, bsxfun(@rdivide, SbOrthTmp, SwOrthTmp), reshape(numbK,1,[],K)),3); 
                if (JLDAOrth>maxJLDAOrth)
                    maxJLDAOrth = JLDAOrth;
                    bestDir(:,2) = dirVecOrth;
                end
            end
        end
        if (bestDirLast == bestDir)
            break;
        end 
    end
    meanHandLDA = bestDir' * meanHandRGB;
    covHandLDA = bestDir' * covHandRGB * bestDir;
    if (mode==2)
        weightHandLDA = maxJLDA / maxJLDAOrth;
    else
        weightHandLDA = 0;
    end

end


% Create on Dec, 2016
% @author: zhongnan qu
function [ piGmm, meanGmm, covGmm ] = emGmmKmeans( data, K )
%   This function is used to estimate the K-component GMM parameters of 
%   training data based on the EM algorithm. The first step of estimation  
%   is clustering the training data into K clusters using K-means.
%   Input: training data (2D table with numbers of observation), number of
%   components.
%   Output: weight of components, mean vector of components, covariance
%   matrix of components

    % define initial convergence criterion of K-means
    % last criterion value
    JKmeansLast = 2E8;
    % new criterion value
    JKmeans = 1E8;
    % reshapethe training data
    dataArray = data(:);
    % descent sort the training data, get sorted data and sorted index  
    [dataResh, index] = sort(dataArray,'descend');
    % remove the unvalid data
    dataResh(dataResh<=0) = [];
    % get the index of all valid data
    index = index(1:length(dataResh));
    % total number of observation 
    N = sum(dataResh);
    
    % convert linear indices to subscripts
    [indexRow, indexCol] = ind2sub(size(data),index);
    % transposition
    x = [indexRow';indexCol'];
    % the dimension of training data
    dimEM = size(x,1);
    % the number of training data (not the number of observation)
    numbEM = size(x,2);
    % set the K largest training data as initial clusters' mean
    initCluster = [indexRow(1:K)';indexCol(1:K)']; 
    meanK = initCluster;
    % set the initial covariance to zero
    covK = zeros(dimEM,dimEM,K);
    % the observation number of each training data in each cluster , size: K*numb
    sumObser = zeros(K,numbEM);
    
    % K-means loop
    while abs(JKmeansLast - JKmeans) >= 1E-7
        %%% E-step %%%
        % calculate all euclidean distances between each training data and each clusters' mean 
        euc_dist_quad_k = reshape(sum(bsxfun(@minus, reshape(x,dimEM,numbEM,[]), reshape(meanK,dimEM,[],K)).^2), numbEM, K);
        % find the minimum euclidean distance of each training data
        % set this training data belonging to this cluster
        [~,w] = min(euc_dist_quad_k,[],2);
        % set the last criterion
        JKmeansLast = JKmeans;
        % define an array
        k = 1:K;
        
        %%% M-step %%%
        % assign the sumObser matrix
        sumObser = bsxfun(@times, bsxfun(@eq,w,k), dataResh)'; 
        % calculate the mean of each cluster
        meanKtmp = sum(bsxfun(@times, reshape(x',numbEM,dimEM,[]), reshape(sumObser',numbEM,[],K)));
        meanK = bsxfun(@rdivide, reshape(meanKtmp, dimEM, K), sum(sumObser,2)');
        % calculate the training data after minusing the mean
        zeroMean = bsxfun(@minus, reshape(x,dimEM,numbEM,[]), reshape(meanK,dimEM,[],K));
        % calculate the weighted training data after minusing the mean
        zeroMeanTmp = bsxfun(@times, zeroMean, reshape(sumObser',[],numbEM,K));
        % calculate the covariance 
        % this covariance calculation only suitable for 2 dimension
        for i = 1:K
            covK(1,1,i) = zeroMean(1,:,i)*zeroMeanTmp(1,:,i)';
            covK(2,2,i) = zeroMean(2,:,i)*zeroMeanTmp(2,:,i)';
            covK(1,2,i) = zeroMean(1,:,i)*zeroMeanTmp(2,:,i)';
            covK(2,1,i) = zeroMean(2,:,i)*zeroMeanTmp(1,:,i)';
        end
        covK = bsxfun(@rdivide, covK, reshape(sum(sumObser,2),1,1,K));
        % calculate the new criterion 
        JKmeansTmp = sum(covK,3);
        JKmeans = trace(JKmeansTmp);
    end
      
    % EM algorithmus for GMM
    % Input: training data, K, initial pi, initial mean, initial covariance
    % Output: estimated pi, estimated mean, estimated covariance
    % initialization of weight of components, size: 1*K
    piGmm = sum(sumObser')/N; 
    % initialization of mean vector of components, size: dim*K
    meanGmm = meanK;
    % initialization of covariance matrix of components, size: dim*dim*K
    covGmm = covK;
    % the responsibility table, p(wk|x(i),theta)
    p = zeros(numbEM, K);
    % the probability of each pixel in each component, N(x(i)|mu(k),sigma(k))
    gaussian = zeros(numbEM, K);
    % the numerator of responsibility, the weighted probability of each 
    % pixel in each component 
    pK = zeros(numbEM, K);
    
    % define the initial log likelihood
    % the last log likelihood
    likelihood_last = 0;
    % the new log likelihood
    likelihood = 10;
    % EM loop
    while (abs((likelihood-likelihood_last)/likelihood)>1E-7)
        %%% E-step %%%
        % set the last log likelihood
        likelihood_last = likelihood;
        % calculate all N(x(i)|mu(k),sigma(k))
        for i = 1:numbEM
            for j = 1:K
                gaussian(i,j) = mvnpdf(x(:,i),meanGmm(:,j),covGmm(:,:,j));
            end
        end
        % calculate all p(wk|x(i),theta)
        pK = bsxfun(@times, piGmm,gaussian);
        pSum = sum(pK,2);
        p = bsxfun(@rdivide, pK, pSum);
        
        %%% M-step %%%
        % calculate the resposibility of each training data (weighted with 
        % the number of observation), size: numb*K
        pPixel = bsxfun(@times, p, dataResh); 
        % get the total responsibility of each component
        nK = sum(pPixel,1);
        % calculate the mean vector of each component
        meanGmmTmp1 = bsxfun(@times, reshape(x',numbEM,dimEM,[]), reshape(pPixel,numbEM,[],K));
        meanGmmTmp = reshape(sum(meanGmmTmp1,1),dimEM,K);
        meanGmm =  bsxfun(@rdivide, meanGmmTmp, nK);
        % calculate the weight of each component
        piGmm = nK/N;
        % calculate the training data after minusing the mean vector 
        zeroMeanGmm = bsxfun(@minus, reshape(x,dimEM, numbEM,[]), reshape(meanGmm,dimEM,[],K)); %dim*numb*K
        % calculate the weighted training data after minusing the mean
        % vector
        zeroMeanGmmTmp = bsxfun(@times,zeroMeanGmm,reshape(pPixel,[],numbEM,K));
        % calculate the covariance matrix of each component
        % this covariance calculation only suitable for 2 dimension
        % set the positive difined flag to true
        ifPosDef = 1;
        for i = 1:K
            covGmm(1,1,i) = zeroMeanGmm(1,:,i)*zeroMeanGmmTmp(1,:,i)';
            covGmm(2,2,i) = zeroMeanGmm(2,:,i)*zeroMeanGmmTmp(2,:,i)';
            covGmm(1,2,i) = zeroMeanGmm(1,:,i)*zeroMeanGmmTmp(2,:,i)';
            covGmm(2,1,i) = covGmm(1,2,i);
            % check if each covariance matrix is positive difined
            if(det(covGmm(:,:,i))<0.001)
                ifPosDef = 0;
            end
        end
        covGmm = bsxfun(@rdivide, covGmm, reshape(nK,1,[],K));
        % calculate the new log likelihood
        likelihood = sum(bsxfun(@times, dataResh, log(pSum)));
        % if there is a non-positive defined covariance matrix, calculate
        % the ratio of the relative variation of log likelihood
        if (ifPosDef==0)
            disp((likelihood-likelihood_last)/likelihood);
            % jump out of the loop
            break;
        end
    end    
end


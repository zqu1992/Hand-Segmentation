function [ w, meanK, covK ] = kmeans( data, K )
% The inputs to Exercise3 kmeans.m are the motion data, the initial cluster label and the number of
% clusters. The outputs of both function are the 3 plots required in Exercise3.a ? b).
    J_last = 2E8;
    J = 1E8;
    dataArray = data(:);
    [dataResh, index] = sort(dataArray,'descend');
    
    dataResh(dataResh<=0) = [];
    N = sum(sum(dataResh));
    index = index(1:length(dataResh));
    
    [indexRow, indexCol] = ind2sub(size(data),index);
    init_cluster = [indexRow(1:K)';indexCol(1:K)']; 
    meanK = init_cluster;
    x = [indexRow';indexCol'];
    covK = zeros(size(x,1),size(x,1),K);
    w = zeros(size(x,2),1);
    while (J_last - J) >= 1E-6

        euc_dist_quad_k = reshape(sum(bsxfun(@minus, reshape(x,size(x,1),size(x,2),[]), reshape(meanK,size(x,1),[],K)).^2), size(x,2), K)';
        [euc_dist_quad,w] = min(euc_dist_quad_k);
        J_last = J;
        k = 1:K;
        
        sumObser = bsxfun(@times, bsxfun(@eq,w,k'), dataResh'); % each pixel in which cluster and has how many observations
        meanKtmp = sum(bsxfun(@times, reshape(x',size(x,2),size(x,1),[]), reshape(sumObser',size(x,2),[],2)));
        meanK = bsxfun(@rdivide, reshape(meanKtmp, K, size(x,1))', sum(sumObser,2)')';
        zeroMean = bsxfun(@minus, reshape(x,size(x,1),size(x,2),[]), reshape(meanK,size(x,1),[],K));
        zeroMeanTmp = bsxfun(@times, zeroMean, reshape(sumObser',[],size(x,2),K));
        
        for i = 1:K
            covK(1,1,i) = zeroMean(1,:,i)*zeroMeanTmp(1,:,i)';
            covK(2,2,i) = zeroMean(2,:,i)*zeroMeanTmp(2,:,i)';
            covK(1,2,i) = zeroMean(1,:,i)*zeroMeanTmp(2,:,i)';
            covK(2,1,i) = covK(1,2,i);
        end
        
        covK = bsxfun(@rdivide, covK, reshape(sum(sumObser,2),1,1,K));
        JTmp = sum(covK,3);
        J = trace(JTmp);
    end
      
    
end


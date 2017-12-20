function [ pi_gmm, mean_gmm, cov_gmm ] = emGMM( data, K )
    % EM algorithmus for GMM
    % Input: training data, K, initial pi, initial mean, initial covariance
    % Output: learned pi, learned mean, learned covariance
    
    % total number of observation 
    n = sum(sum(data));
    
    % p(wk|x(i),theta)
    p = zeros(size(data,1), size(data,2), K); 
    gaussian = zeros(size(data,1), size(data,2), K);
    p_k = zeros(size(data,1), size(data,2), K);
    p_sum = zeros(size(data,1), size(data,2));
    [index_row, index_col] = find(data>0);
    index = [index_row'; index_col'];
    
    pi_gmm = pi_init;
    mean_gmm = mean_init;
    cov_gmm = cov_init;

    likelihood_last = 0;
    likelihood = 10;
    
    while (abs((likelihood-likelihood_last)/likelihood)>0.0000001)
        likelihood_last = likelihood;
        
        %%% E-step %%%
        % calculate all N(x(i)|mu(k),sigma(k))
        for i = 1:length(index)
            for k = 1:K
                gaussian(index(1,i),index(2,i),k) = mvnpdf(index(:,i),mean_gmm(:,k),cov_gmm(:,:,k));
            end
        end
        % calculate all p(wk|x(i),theta)
        for i = 1:length(index)
            for k = 1:K
                p_k(index(1,i),index(2,i),k) = pi_gmm(k) * gaussian(index(1,i),index(2,i),k);
                p_sum(index(1,i),index(2,i)) = p_sum(index(1,i),index(2,i)) + p_k(index(1,i),index(2,i),k);
            end
            for k = 1:K
                p(index(1,i),index(2,i),k) = p_k(index(1,i),index(2,i),k) / p_sum(index(1,i),index(2,i));
            end
        end

        %%% M-step %%%
        for k = 1:K
            p_pixel = data .* p(:,:,k);
            nk = sum(sum(p_pixel));
            mean_gmm(:,k) = [0;0];
            for i = 1:length(index) 
                mean_gmm(:,k) = mean_gmm(:,k) + [index(1,i);index(2,i)] * p_pixel(index(1,i),index(2,i));
            end
            mean_gmm(:,k) = mean_gmm(:,k) / nk;
            cov_gmm(:,:,k) = [0 0;0 0];
            for i = 1:length(index)
                cov_gmm(:,:,k) = cov_gmm(:,:,k) + p_pixel(index(1,i),index(2,i)) * ([index(1,i);index(2,i)] - mean_gmm(:,k)) * ([index(1,i);index(2,i)] - mean_gmm(:,k))';
            end
            cov_gmm(:,:,k) = cov_gmm(:,:,k) / nk;
            pi_gmm(k) = nk/n;
        end
        likelihood = 0;
        for i = 1:length(index)
            likelihood = likelihood + data(index(1,i),index(2,i))*log(p_sum(index(1,i),index(2,i)));
        end
    end


end


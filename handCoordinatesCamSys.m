% load the depth data of hand from AdaptiveHandDetection.m
load('handDepth.mat')
% remove the coordinates with -Inf or Inf value (error when transfer)
handPtCamSysReal = handPtCamSys(sum(abs(handPtCamSys),2)<30, :);
% calculate the mean coordinates
handPtCamSysMean = mean(handPtCamSysReal,1);
% remove the points which are far away from the mean point
% recalculate the mean coordinates
handPtCamSysMeanNew = mean(handPtCamSysReal(sum(abs(bsxfun(@minus,handPtCamSysReal,handPtCamSysMean)),2)<1,:),1);
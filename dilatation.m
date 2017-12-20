% Create on Dec, 2016
% @author: zhongnan qu
function [ dilatForeground ] = dilatation( foreground, structuralElement, row, col)
%   This function is used to calculate the Minkowski-addition (dilatation).
%   Input: foreground coordinates (size: dim*numb) need to be extended, 
%   structure element, the limit value of image row, the limit value of 
%   image column 
%   Output: dilatated result coordinates
    
    % this step can be deleted 
    structuralElement = -structuralElement;
    % dimension of dataset
    dim = size(foreground, 1);
    % number of pixels in foreground
    numbFG = size(foreground, 2);
    % number of structure element
    numbSE = size(structuralElement,2);
    % all result pixels of every translation
    dilatTmp = bsxfun(@plus, reshape(foreground,dim,numbFG,[]),reshape(structuralElement,dim,[],numbSE));
    % find the union set of all these result pixels
    dilatForeground = dilatTmp(:,:,1)';
    for i = 2:numbSE
        dilatForeground = union(dilatForeground, dilatTmp(:,:,i)', 'rows');
    end
    % transposition 
    dilatForeground = dilatForeground';
    % get the subscripts of all coordinates after dilatation
    dilatFGRow = dilatForeground(1,:);
    dilatFGCol = dilatForeground(2,:);
    % set the value over limit to limit
    dilatFGRow(dilatFGRow>row) = row;
    dilatFGCol(dilatFGCol>col) = col;
    dilatFGRow(dilatFGRow<1) = 1;
    dilatFGCol(dilatFGCol<1) = 1;
    % return the result
    dilatForeground = [dilatFGRow;dilatFGCol];
end


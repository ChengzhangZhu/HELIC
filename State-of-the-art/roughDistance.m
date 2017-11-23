function [distance] = roughDistance(data)
%Usage: [distance] = roughDistance(data)
%Reference: Cao, F., Liang, J., Li, D., Bai, L. & Dang, C. 2012, 'A dissimilarity measure for the k-Modes clustering algorithm', Knowledge-Based Systems, vol. 26, pp. 120¨C7.
%
%Input: data, the original data n*m, n is the number of data, m is the
%number of attribute
%Output: similarity, m*1 cell with k*k matrix in each cell, where k is the number of attribute value in each attribute.


%% initial variable
distance = cell(1,size(data,2));

%% calculate value pair distance in each attribute
for j = 1 : size(data,2)
    attribute = data(:,j); % j-th attribute
    attriValue = unique(attribute); % unique attribute value in j-th attribute
    numOfValue = length(attriValue); % number of unique attribute value in j-th attribute
    disMatrix = ones(numOfValue, numOfValue); %initial distance matrix for each value pair
    for ii = 1:numOfValue  % calculate distance for each value pair
        disMatrix(ii,ii) = 1 - 1/length(find(attribute == attriValue(ii)));
    end
    distance{j} = disMatrix;
end
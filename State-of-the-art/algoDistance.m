function [distance] = algoDistance(data)
%Usage: [distance] = algoDistance(data)
%Reference: Amir Ahmad, Lipika Dey. 'A method to compute distance between
%two categorical values of same attribute in unsupervised learning for
%categorical data set', Pattern Recognition Letters. 28(2007):110-118
%
%Input: data, the original data n*m, n is the number of data, m is the
%number of attribute
%Output: distance, m*1 cell with k*k matrix in each cell, where k is the number of attribute value in each attribute.

%% initial variable
distance = cell(1,size(data,2));

%% calculate value pair distance in each attribute
for j = 1 : size(data,2)
    attribute = data(:,j); % j-th attribute
    attriValue = unique(attribute); % unique attribute value in j-th attribute
    numOfValue = length(attriValue); % number of unique attribute value in j-th attribute
    disMatrix = zeros(numOfValue, numOfValue); %initial distance matrix for each value pair
    for ii = 1:numOfValue -1  % calculate distance for each value pair
        for jj = ii+1:numOfValue
            disSum = 0; %initial pair distance regarding to each attribute
            for oj = 1:size(data,2)  % calculate distance for each value pair regarding to oj-th attribute, where oj~=j
                if oj~=j
                    disSum = disSum+delta(attriValue(ii),attriValue(jj),attribute,data(:,oj)); %call delta function to calculate distance
                end
            end
            disSum = disSum/(size(data,2)-1); % calculate the average distance value for each attribute. (We can change the weight of attribute for extending)
            disMatrix(ii,jj) = disSum;
            disMatrix(jj,ii) = disSum;
        end
    end
    distance{j} = disMatrix;
end

%% define delta function
function dis = delta(value1, value2, attribute1, attribute2)
%% initial variable
attriValue2 = unique(attribute2); %the unique attribute value of attribute 2
w1 = []; % initial subset w
w2 = []; % initial subset ~w
pw1 = 0; % initial conditional probability in subset w
pw2 = 0; % initial conditional probability in subset ~w
%% finding the maximizing subset w and calculate the probability in subset w (pw1) and ~w (pw2)
for i = 1 : length(attriValue2) % find subset of value of attribute 2
    p21 = length(find(attribute1==value1&attribute2==attriValue2(i)))/length(find(attribute1 == value1)); % calculate the conditional probability 
    p22 = length(find(attribute1==value2&attribute2==attriValue2(i)))/length(find(attribute1 == value2)); % calculate the conditional probability 
    if p21 > p22
        w1 = [w1 attriValue2(i)];
        pw1 = pw1 + p21;
    else
        w2 = [w2 attriValue2(i)];
        pw2 = pw2 + p22;
    end
end
%% calculate the distance value
dis = pw1 + pw2 -1;









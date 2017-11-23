function [distance] = DILCA(data)
%Usage: [distance] = DILCA(data)
%Reference: Ienco, D., Pensa, R.G. & Meo, R. 2012, 'From Context to Distance: Learning Dissimilarity for Categorical Data Clustering', ACM Transactions on Knowledge Discovery from Data, vol. 6, no. 1, pp. 1šC25.
%Here we choice the non paramatric method to select context
%Input: data, the original data n*m, n is the number of data, m is the
%number of attribute
%Output: distance, m*1 cell with k*k matrix in each cell, where k is the number of attribute value in each attribute.

%% initial variable
distance = cell(1,size(data,2));

%% calculate SU matrix
SU = zeros(size(data,2),size(data,2));
for i = 1:size(data,2)
    for j = 1:size(data,2)
        if i~=j
            SU(i,j) = ComputeSU(data(:,i),data(:,j));
        end
    end
end

%% calculate the context
context = cell(1,size(data,2));
for j = 1 : size(data,2)
    erase = [];
    eraseSu = [];
    su1 = SU(j,:);
    [~, suIndex] = sort(su1,'descend');
    for i = suIndex
        if isempty(find(eraseSu == i, 1))
            if i~=j
                su2 = SU(i,:);
                for ii = suIndex(find(suIndex==i):end)
                    if ii~=j
                        if su2(ii)>=su1(ii)
                            erase = [erase ii];
                            eraseSu = [eraseSu ii];
                        end
                    end
                end
            end
        end
    end
contextJ = 1:size(data,2);
contextJ(erase) = [];
context{j} = contextJ;
end

%% calculate value pair distance in each attribute
for j = 1 : size(data,2)
    attribute = data(:,j); % j-th attribute
    attriValue = unique(attribute); % unique attribute value in j-th attribute
    numOfValue = length(attriValue); % number of unique attribute value in j-th attribute
    disMatrix = zeros(numOfValue, numOfValue); %initial distance matrix for each value pair
    for ii = 1:numOfValue -1  % calculate distance for each value pair
        for jj = ii+1:numOfValue
            disSum = 0; %initial pair distance regarding to each attribute
            for oj = context{j}  % calculate distance for each value pair regarding to context oj, where oj~=j
                if oj~=j
                    disSum = disSum+inter(attriValue(ii),attriValue(jj),attribute,data(:,oj)); %call inter function to calculate distance
                end
            end
            disSum = disSum/(length(context)); % calculate the average distance value for each attribute. (We can change the weight of attribute for extending)
            disMatrix(ii,jj) = sqrt(disSum);
            disMatrix(jj,ii) = disMatrix(ii,jj);
        end
    end
    distance{j} = disMatrix;
end

%% define SU computation function
function su = ComputeSU(attribute1, attribute2)
attriValue1 = unique(attribute1);
attriValue2 = unique(attribute2);
%% calculate the entropy
H1 = 0;
H2 = 0;
for i = 1:length(attriValue1)
    p1 = length(find(attribute1 == attriValue1(i)))/length(attribute1);
    H1 = H1 - p1*log2(p1);
end
for i = 1:length(attriValue2)
    p2 = length(find(attribute2 == attriValue2(i)))/length(attribute2);
    H2 = H2 - p2*log2(p2);
end

%% calculate the conditional entropy
H3 = 0; % the conditional entropy
for i = 1:length(attriValue2)
    subAttri = attribute1(attribute2 == attriValue2(i)); %construct subset of attribute 1 under attribute 2's i-th value
    p2 = length(find(attribute2 == attriValue2(i)))/length(attribute2);
    subValue = unique(subAttri);
    for j = 1 : length(subValue)
        p3 = length(find(subAttri == subValue(j)))/length(subAttri);
        H3 = H3 - p2*p3*log2(p3);
    end
end

%% calculate the information gain
IG = H1 - H3;
%% calculate the symmetrical uncertainty
su = 2*IG/(H1+H2);

%% define inter function
function dis = inter(value1, value2, attribute1, attribute2)
dis = 0;
attriValue2 = unique(attribute2); %the unique values in attribute 2
for i = 1:length(attriValue2)
    subAttri = attribute1(attribute2 == attriValue2(i)); %construct subset of attribute 1 under attribute 2's i-th value
    p1 = length(find(subAttri == value1))/length(subAttri);
    p2 = length(find(subAttri == value2))/length(subAttri);
    dis = dis + (p1-p2)^2;
end


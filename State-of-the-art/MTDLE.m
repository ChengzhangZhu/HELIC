function [distance,lowLevelDis] = MTDLE(data, label)
%Usage: [distance,lowLevelDis] = MTDLE(data, label)
%This code is the implement of the following paper:
%"Zhang, K., Wang, Q., Chen, Z., Marsic, I., Kumar, V., Jiang, G. & Zhang, J. 2015, 'From categorical to numerical: Multiple transitive distance learning and embedding', SIAMInternational Conference on Data Mining, pp. 46šC54."
%Input: data, the categorical data matrix n * d
%          label, the corresponding label n*1
%Output: distance, the distance matrix
%            lowLevelDis, the low level attribute value distance and
%            combination coefficience.
[D1,V1] = objDisCalc(data, 4); %cosine distance
[D2,V2] = objDisCalc(data, 2); %normalized co-occurence
% [D3,V3] = objDisCalc(data, 3); %mutual information
n = size(data,1);
interIndex = 1;
intraIndex = 1;
for i = 1 : n
    for j = 1 : n
        index(i,j) = label(i) == label(j);
    end
end
numOfItra = sum(sum(index));
numOfInter = n*n - numOfItra;
% interD = zeros(numOfInter,3);
% intraD = zeros(numOfItra,3);
interD = zeros(numOfInter,2);
intraD = zeros(numOfItra,2);
for i = 1 : n
    for j = 1 : n
        if label(i) ~= label(j)
%             interD(interIndex,:) = [D1(i,j) D2(i,j) D3(i,j)];
            interD(interIndex,:) = [D1(i,j) D2(i,j)];
            interIndex = interIndex + 1;
        else
%             intraD(intraIndex,:) = [D1(i,j) D2(i,j) D3(i,j)];
            intraD(intraIndex,:) = [D1(i,j) D2(i,j)];
            intraIndex = intraIndex + 1;
        end
    end
end
A = intraD'*intraD;
B = interD'*interD;
cvx_begin
% variable a(3,1)
% variable b(3,1)
variable a(2,1)
variable b(2,1)
minimize(b'*(B.^(-0.5)*A*B.^(-0.5))*b);
subject to
b == B.^(-0.5)*a;
a >= 0;
% b'*B.^(-0.5)*ones(3,1) == 1;
b'*B.^(-0.5)*ones(2,1) == 1;
cvx_end
lowLevelDis.ValueDis{1} = V1;
lowLevelDis.ValueDis{2} = V2;
% lowLevelDis.ValueDis{3} = V3;
lowLevelDis.Coeff = a;

% distance = a(1).*D1 +a(2).*D2 + a(3).*D3;
distance = a(1).*D1 +a(2).*D2;

function [objDis,valueDis] = objDisCalc(data, model)
%calculate the inter-object distaince
valueDis = valueDisCalc(data, model);
n = size(data,1);
d = size(data,2);
objDis = zeros(n,n);
for i = 1 : n 
    disp(['is for', int2str(i),'-th object'])
    for j = i+1 : n
        disij = 0;
        for k = 1 : d
            disMatrix = valueDis{k};
            disij = disij + disMatrix(data(i,k),data(j,k));
        end
        objDis(i,j) = disij;
        objDis(j,i) = objDis(i,j);
    end
end

function valueDis = valueDisCalc(data, model)
%calculate the intra-attribute value distance
graph = dPartiteGraph(data, model); %the d-parptite graph of data
d = size(data,2); %the dimension of data
numIndex = zeros(d+1,1); %the index record the number of value in each attribute
for i = 1 : d
    value = unique(data(:,i));
    numIndex(i+1) = length(value);
end
valueDis = cell(d,1);
for i = 1 : d
    value = unique(data(:,i));
    numOfValue = length(value);
    disMatrix = zeros(numOfValue, numOfValue);
    for ii = 1 : numOfValue
        for jj = ii + 1 : numOfValue
            G = graph;
            G(graph == realmax) = 0;
            disMatrix(ii,jj) = graphshortestpath(sparse(G),ii+sum(numIndex(1:i)), jj+sum(numIndex(1:i)));
            disMatrix(jj,ii) = disMatrix(ii,jj);
        end
    end
    valueDis{i} = disMatrix; 
end


function graph = dPartiteGraph(data, model)
%construct d-parptite graph for finding the shortest path
d = size(data,2); %the dimension of data
numIndex = zeros(d+1,1); %the index record the number of value in each attribute
for i = 1 : d
    value = unique(data(:,i));
    numIndex(i+1) = length(value);
end
totalValueNum = sum(numIndex);
graph = realmax*ones(totalValueNum, totalValueNum);
for i = 1 : d
    value = unique(data(:,i));
    numOfValue = length(value);
    for j = 1 : numOfValue
        for ii = i + 1 : d
            valueCo = unique(data(:,ii));
            numOfValueCo = length(valueCo);
            for jj = 1 : numOfValueCo
                graph(j+sum(numIndex(1:i)),jj+sum(numIndex(1:ii))) = coOccurrence(data,i,j,ii,jj,model); %calculate the co-occurrence of two attribute;
                graph(jj+sum(numIndex(1:ii)), j+sum(numIndex(1:i))) = graph(j+sum(numIndex(1:i)),jj+sum(numIndex(1:ii)));
            end   
        end
    end
end

function co = coOccurrence(data,i,j,ii,jj,model)
%calculate different co-occrrence value
%data is the categorical data matrix
%i,j,ii,jj are the position index for the i-th attribute j-th value and ii-the attribute jj-th value
% model is used to control the type of co-occurrence measure: 1.
% co-occurrence; 2. normalized co-occurrence...

if model == 1 %co-occurrence
    value1 = data(j,i);
    value2 = data(jj,ii);
    occ1 = data(:,i) == value1;
    occ2 = data(:,ii) == value2;
    co = sum(occ1&occ2);
    co = log(-co);
end

if model == 2 %normalized co-occurrence
    value1 = data(j,i);
    value2 = data(jj,ii);
    occ1 = data(:,i) == value1;
    occ2 = data(:,ii) == value2;
    co = sum(occ1&occ2)/sum(occ1|occ2);
    co = -log(co);
end

if model == 3 % mutual information
    value1 = data(j,i);
    value2 = data(jj,ii);
    occ1 = data(:,i) == value1;
    occ2 = data(:,ii) == value2;
    p1 = sum(occ1)/size(data,1);
    p2 = sum(occ2)/size(data,1);
    p12 = sum(occ1&occ2)/size(data,1);
    if p12 ~= 0
        co = p12*log(p12/(p1*p2));
    else
        co = 0;
    end
    co = -log(co);
end

if model == 4 %cosine distance
    value1 = data(j,i);
    value2 = data(jj,ii);
    occ1 = data(:,i) == value1;
    occ2 = data(:,ii) == value2;
%     co = double(occ1)'*double(occ2)/sqrt(norm(double(occ1))*norm(double(occ2)));
   co = pdist2(double(occ1)',double(occ2)','cosine');
end

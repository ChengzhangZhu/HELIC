function [W,D, kernels] = HELIC_GD(data, label)
%Usage: [W,D, kernels] = HELIC_GD(data, label)
numericalSpace = dataMapping(data, label);
kernels = produceKernel(numericalSpace);
[W,D] = cMKL(data, kernels, label);

function numericalSpace = dataMapping(categoricalData, label)

d = size(categoricalData,2);
numericalSpace = cell(d,1);
for i = 1 : d
    value = unique(categoricalData(:,i));
    numOfValue = length(value);
    for ii = 1 : numOfValue
        intra = intraMap(value(ii),categoricalData(:,i));
        inter = interMap(value(ii), categoricalData, i);
        attriClass = attriClassMap(value(ii), categoricalData, i, label);
%         match = matchMap(value(ii),categoricalData(:,i));
%         numericalSpace{i}(ii,:) = [intra; inter; attriClass; match]'; %can give weight to each dimension, here is the uniform weight version
        numericalSpace{i}(ii,:) = [intra; inter; attriClass]'; %can give weight to each dimension, here is the  weight version
    end
end

function intra = intraMap(value, attribute)
intra = sum(value==attribute)/length(attribute);

function inter = interMap(value, categoricalData, attributeIndex)
coAttribute = categoricalData;
coAttribute(:,attributeIndex) = [];
numOfCoValue = length(unique(coAttribute));
clear coAttribute;
inter = zeros(numOfCoValue, 1);
valueIndex1 = value == categoricalData(:,attributeIndex);
for i = 1 : size(categoricalData,2)
    if i ~=attributeIndex
        coValue = unique(categoricalData(:,i)); 
        for j = 1 : length(coValue)          
            valueIndex2 = coValue(j) == categoricalData(:,i);
            inter(j) = sum(valueIndex1 & valueIndex2)/sum(valueIndex1);%define inter mapping function, this is the conditional probability version w.r.t. compared value
        end
    end
end

function attriClass = attriClassMap(value, categoricalData, attributeIndex, label)
classValue = unique(label);
numOfClass = length(classValue);
attriClass = zeros(numOfClass, 1);
valueIndex1 = value == categoricalData(:,attributeIndex);
for i = 1 : numOfClass
    valueIndex2 = classValue(i) == label;
    attriClass(i) = sum(valueIndex1 & valueIndex2)/sum(valueIndex1);
end

function match = matchMap(value, attribute)
attriValue = unique(attribute);
numOfValue = length(attriValue);
match = zeros(numOfValue,1);
match(value == unique(attribute)) = 1;

function kernels = produceKernel(numericalSpace)
kernels  = cell(length(numericalSpace)*23,1);
for ii = 1 : length(numericalSpace)
for i = 1 : 1 : 20
    kernels{(ii-1)*23+i} = kernel_matrix(numericalSpace{ii},'RBF_kernel',2^(i-11));
end
% for i = 31:1:50
%     KH(:,:,i) = kernel_matrix(data,'wav_kernel',[1/2^(i-10),1/2^(i-10),i]);
% end
for i = 21:1:23
        kernels{(ii-1)*23+i} =  kernel_matrix(numericalSpace{ii},'poly_kernel',[0,i-20]);
end

end

function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);

if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega.*kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega.*kernel_pars(1));
    end
    
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end
omega = omega./sqrt(diag(omega)*diag(omega)');
% omega = omega/trace(omega);

function [W,d] = cMKL(data, kernels, label)
%Usage [W,d] = cMKL(data, kernels, label)
numOfKernel = length(kernels);
numOfValuePerKernel  = zeros(numOfKernel,1);
numOfWeight = 0;
numOfObject = length(label);
beginPoint = ones(numOfKernel+1,1);
for i = 1 : numOfKernel
    numOfWeight = numOfWeight + size(kernels{i},1);
    numOfValuePerKernel(i) = size(kernels{i},1);
    beginPoint(i+1) = beginPoint(i) + numOfValuePerKernel(i);
end
numOfDiffKernel = numOfKernel/size(data,2);

%% construct y
% y = ones(length(label),length(label));
% for i = 1 : length(label)
%     for j = 1: length(label)
%         if label(i)~=label(j)
%             y(i,j) = -1;
%         end
%     end
% end

%% construct K Square and Y
Y = zeros((numOfObject^2+numOfObject)/2,1);
K = zeros((numOfObject^2+numOfObject)/2,numOfWeight);
indexK = 1;
for i = 1 : numOfObject
    for j =i+1 : numOfObject
         kVector = [];
            for ii = 1 : numOfKernel
                attrIndex = ceil(ii/numOfDiffKernel);
                kVector= [kVector (kernels{ii}(:,data(i,attrIndex)) - kernels{ii}(:,data(j,attrIndex)))'];
            end
            K(indexK,:) = kVector;
            %construct Y
            if label(i) ~= label(j)
                Y(indexK) = 1;
            end
            indexK = indexK + 1;
    end
end
K = K.^2;

%% 
cvx_begin
variable W(numOfWeight);
minimize 0.5*norm(K*W-Y,2) + sum(W)
subject to 
W >= 0;
% sum(W) == 1
cvx_end

indexK = 1;
d = zeros(numOfObject, numOfObject);
for i = 1 : numOfObject
    for j = i+1 : numOfObject
        d(i,j) = K(indexK,:)*W;
        d(j,i) = d(i,j);
        indexK = indexK+1;
    end
end
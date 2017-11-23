function dataPreprocess(data,dataname,numericalIndex,labelIndex)
%Usage: dataPreprocess(data,dataname,numericalIndex,labelIndex)
%
%This function is used to preprocess mixed data
%Input: data, the data set in cell format
%          dataname, data set name in char format
%          numericalIndex, the numerical attribute index in vector format
%          labelIndex, the index of label column

index = 1:size(data,2);
nominalIndex = setdiff(index,numericalIndex);
nomialData = data(:,nominalIndex);
nomialData = categorical(nomialData);
nominalProData = zeros(size(nomialData,1),size(nomialData,2));
numericalData = data(:,numericalIndex);
numericalData = cell2mat(numericalData);
numericalProData = mapminmax(numericalData')';
labelCol = find(nominalIndex == labelIndex);
for dd = 1:size(nomialData,2)
    tempcolumn = double(nomialData(:,dd));
    temp_back = tempcolumn;
    unique_values = unique(tempcolumn);
    
    for uu = 1:length(unique_values)
        ind = (tempcolumn == unique_values(uu));
        temp_back(ind,:) = uu;%unique_values(uu);
    end
    nominalProData(:,dd) = temp_back;
end
label = nominalProData(:,labelCol);
nominalProData(:,labelCol) = [];

index = zeros(20,size(nominalProData,1));
for i = 1 : 20
    flage = 1;
    while flage ==1
        randIndx = randperm(size(nominalProData,1));
        trnIndex = find(randIndx > ceil(size(nominalProData,1)/3));
        yapp = label(randIndx > ceil(size(nominalProData,1)/3));
        if length(unique(yapp)) == length(unique(label))
            flage = 0;
        end
    end
    index(i,:) = randIndx;
end
save(dataname,'data','label','nomialData','nominalProData','numericalData','numericalProData','numericalIndex','nominalIndex','labelIndex','index');
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:25:53 2017

@author: Chengzhang Zhu
"""
import scipy.io as scio
import tensorflow as tf
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt  

#initial functions
def weight_variable(shape, name = 'weight'):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)
        
def bias_variable(shape, name = 'bias'):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

def numOfAttriValue(data):
    numOfValue = np.zeros([data.shape[1]], dtype = 'int32')
    for i in range(data.shape[1]):
        numOfValue[i] = len(np.unique(data[:,i]))
    return numOfValue
    
def concatDict(dictionaryList):
    dictionary = dict()
    numOfAttri = len(dictionaryList[0])
    numOfDict = len(dictionaryList)
    for i in range(numOfAttri):
        mapping = dictionaryList[0][i]
        for j in range(numOfDict - 1):
            j = j + 1
            mapping = np.concatenate([mapping, dictionaryList[j][i]], axis = 1)
        dictionary[i] = mapping
    return dictionary

def VDMEmbed(data, label):
    dictionary = dict()
    uniLabel = np.unique(label)
    for attri in range(data.shape[1]):
        values = np.unique(data[:,attri])
        embed = np.zeros([len(values), len(uniLabel)], dtype = 'float32')
        for value in range(len(values)):
            for c in range(len(uniLabel)):
                prob = np.size(np.where(data[np.where(label[:,0] == uniLabel[c]),:] == values[value]))/data.shape[0]
                embed[value,c] = prob
        dictionary[attri] = embed
    return dictionary

def intraAttriEmbed(data):
    dictionary = dict()
    for attri in range(data.shape[1]):
        values = np.unique(data[:,attri])
        embed = np.zeros([len(values), 1], dtype = 'float32')
        for value in range(len(values)):
            embed[value, 0] = np.size(np.where(data[:,attri] == values[value]))/data.shape[0] 
        dictionary[attri] = embed
    return dictionary
    
def interAttriEmbed(data):
    dictionary = dict()
    numOfValue = numOfAttriValue(data)
    for attri in range(data.shape[1]):
        values = np.unique(data[:,attri])
        embed = np.zeros([len(values), np.sum(numOfValue) - len(values)], dtype = 'float32')
        for value in range(len(values)):
            index = 0
            for contextAttri in np.delete(np.r_[0:data.shape[1]], attri):
                contextValues = np.unique(data[:,contextAttri])
                for contextValue in contextValues:
                    embed[value, index] = np.size(np.where(data[np.where(data[:,contextAttri] == contextValue), attri] == values[value]))/np.size(np.where(data[:,contextAttri] == contextValue))
                    index = index + 1
        dictionary[attri] = embed
    return dictionary

def kernel(data, kernelType = 'Guassian', kernelPara = 1):
    nb_data = np.shape(data)[0]
    if kernelType == 'Guassian':
        XXh = np.matmul(np.sum(data**2, 1).reshape([nb_data,1]), np.ones([1,nb_data]))
        omega = XXh + XXh.transpose() - 2*np.matmul(data, data.transpose())
        omega = np.exp(-omega*kernelPara)
    elif kernelType == 'Polynominal':
        omega = (np.matmul(data, data.transpose()) + kernelPara[0])**kernelPara[1]
    else:
        assert('Error kernel type')
    omega = omega/np.sqrt(np.matmul(np.diag(omega), np.diag(omega).transpose()))
    return omega
    
##data generate function
def generate_batch_newInstance(data, label, kernelSpace, batch_size = 20, data_index = 0):
    #assert len(data) % batch_size == 0
    if data_index > data.shape[0] - batch_size:
        data_index = 0
    batch = data[data_index:data_index + batch_size,:]
    batchLabel = label[data_index:data_index + batch_size,:]
    data_index = data_index + batch_size
    numOfObject = batch.shape[0]
    batch_train = list()
    batch_label = list()
    
    for i in range(numOfObject):
        for j in range(numOfObject - i - 1):
            k = i + j + 1
            kVector = []
            for space in kernelSpace:
                kVector = np.concatenate([kVector,space[0][:,batch[i, space[1]] - 1] - space[0][:,batch[k, space[1]] - 1]])
            batch_train.append(kVector)
            if batchLabel[i,0] != batchLabel[k,0]:
                batch_label.append([-1])
            else:
                batch_label.append([1])
    batch_train = np.array(batch_train)
    batch_label = np.array(batch_label)
    batch_train = batch_train ** 2           
    return batch_train, batch_label, data_index

##data generate function
def generate_batch_randomNewInstance(data, label, kernelSpace, batch_size = 20):
    #assert len(data) % batch_size == 0
    batch_train = list()
    batch_label = list()
    index = np.r_[0:len(data)]
    while len(index) < batch_size:
        index = np.concatenate((index,index))
    shuffle(index)
    index1 = index[0:batch_size].copy()
    shuffle(index)
    index2 = index[0:batch_size].copy()
    for i , k in zip(index1, index2):
        kVector = []
        for space in kernelSpace:
#            print(data[i, space[1]] - 1,data[k, space[1]] - 1)
#            print(space[0])
            kVector = np.concatenate([kVector,space[0][:,data[i, space[1]] - 1] - space[0][:,data[k, space[1]] - 1]])
        batch_train.append(kVector)
        if label[i,0] != label[k,0]:
            batch_label.append([-1])
        else:
            batch_label.append([1])
    batch_train = np.array(batch_train)
    batch_label = np.array(batch_label)
    batch_train = batch_train ** 2   
    return batch_train, batch_label

 
def generate_batch(data, label, batch_size = 20, data_index = 0):
    if data_index > data.shape[0] - batch_size:
        data_index = 0
    batch = data[data_index:data_index + batch_size,:]
    batchLabel = label[data_index:data_index + batch_size,:]
    data_index = data_index + batch_size
    return batch, batchLabel, data_index
    
def generate_newInstance(data, label, kernelSpace):
    numOfObject = data.shape[0]
    new_data = list()
    new_label = list()
    for i in range(numOfObject):
        for j in range(numOfObject - i - 1):
            k = i + j + 1
            kVector = []
            for space in kernelSpace:
                kVector = np.concatenate([kVector,space[0][:,data[i, space[1]] - 1] - space[0][:,data[k, space[1]] - 1]])
            new_data.append(kVector)
            if label[i,0] != label[k,0]:
                new_label.append([-1])
            else:
                new_label.append([1])
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    new_data = new_data ** 2
    return new_data, new_label
    
def generate_kernelVector(data, kernelSpace, W, reduce = False):
    numOfObject = data.shape[0]
    kernelVector = list()
    if reduce == True:
        remainList = np.where(W > 1e-3)[0]
        W = W[remainList]
        print(len(W))
    for i in range(numOfObject):
        kVector = []
        for space in kernelSpace:
            kVector = np.concatenate([kVector,space[0][:,data[i, space[1]] - 1]])
        if reduce == True:
            kVector = kVector[remainList]
        kVector = kVector*(W.transpose()**0.5)
        kernelVector.append(kVector[0])
    kernelVector = np.array(kernelVector)        
    return kernelVector
    
def readMatlabData(filename):
    data = scio.loadmat(filename)
    nominalData = data['nominalProData']
    label = data['label']
    return nominalData,label


def HC(data, label):
    HCSpace = concatDict([intraAttriEmbed(data), interAttriEmbed(data), VDMEmbed(data, label)])
    return HCSpace
    
def KernelMapping(HCSpace, kernelPara):
    kernelSpace = list()
    for k in kernelPara:
        for para in kernelPara[k]:
            for subspace in HCSpace:
                kernelSpace.append((kernel(HCSpace[subspace], kernelType = k, kernelPara = para), subspace, k, para))   
    return kernelSpace
    

def readMatlabPartition(filename):
    data = scio.loadmat(filename)
    trainIndex = data['trnIndex']
    tstIndex = data['tstIndex']
    return trainIndex,tstIndex

def oneHotDict(data):
    dictionary = dict()
    for i in range(data.shape[1]):
        dictionaryAttri = dict()
        values = np.unique(data[:,i])
        embeddingSize = len(values)
        baseVector = np.zeros([1, embeddingSize], dtype = 'int32')
        for j in range(embeddingSize):
            embedVector = baseVector.copy()
            embedVector[:,j] = 1
            dictionaryAttri[values[j]] = embedVector
        dictionary[i] = dictionaryAttri
    return dictionary
    
def oneHotEmbedding(data):
    dictionary = oneHotDict(data)
    numOfValue = np.sum(numOfAttriValue(data))
    embedData = np.zeros([data.shape[0], numOfValue])
    bIndex = 0
    for attri in range(data.shape[1]):
        values = np.unique(data[:,attri])
        embeddingSize = len(values)
        eIndex = bIndex + embeddingSize - 1
        for value in values:
            embedData[np.where(data[:,attri] == value), bIndex:eIndex + 1] = dictionary[attri][value][0]
        bIndex = eIndex + 1
    return embedData

def embedFromDict(data, dictionary):
    #check the embedding size
    embedSize = np.zeros([data.shape[1]], dtype = 'int32')
    index = 0
    for dic in dictionary:
        embedSize[index] = dictionary[dic].shape[1]
        index = index + 1
    embedData = np.zeros([data.shape[0], np.sum(embedSize)])
    bIndex = 0
    for attri in range(data.shape[1]):
        eIndex = bIndex + embedSize[attri] - 1
        uniqueValue = np.unique(data[:,attri])
        for value in uniqueValue:
            embedData[np.where(data[:,attri] == value),bIndex:eIndex+1] = dictionary[attri][value-1] #value is int number from 1, which transformed from cactegorical value
        bIndex = eIndex + 1
    return embedData

def useLargeSize(axis,marker_lines = None, fontsize = 'xx-large',fontproperties=None):
    axis.xaxis.get_label().set_size(fontsize)
    axis.yaxis.get_label().set_size(fontsize)
    for label in axis.xaxis.get_ticklabels():
        label.set_fontsize(18)
    for label in axis.yaxis.get_ticklabels():
        label.set_fontsize(18) 
    LW = 2.3
    for line in axis.get_lines():
        line.set_lw( LW )
    leg = axis.get_legend()
    if(leg):
        ltext  = leg.get_texts()  # all the text.Text instance in the legend
        if(fontproperties):
            plt.setp(ltext, fontproperties=fontproperties)        
        plt.setp(ltext, fontsize='x-large')
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        plt.setp(llines,linewidth= LW )
        if(marker_lines and len(marker_lines)>=len(llines)):
            for i in range(0,len(llines)):
                plt.setp(llines[i], 
                    marker = marker_lines[i].get_marker(), 
                    markeredgecolor= marker_lines[i].get_markeredgecolor(),\
                    markerfacecolor= marker_lines[i].get_markerfacecolor(),\
                    markeredgewidth= marker_lines[i].get_markeredgewidth(),
                    markersize= marker_lines[i].get_markersize() )          
                    
#dataName = 'mushroom'
#filename = 'D:\\Experiment\\Couled Multiple Kernel Learning on Data Level\\Coupled Nominal Data for Kernel Learning - v17 non-IID Categorical Data Embedding\\Datasets\\' + dataName + '.mat'
#data, label = readMatlabData(filename)
#import pickle
#pickle.dump(data,open(dataName+'.data','wb'),protocol = 0)

#import pickle
#pickle.dump((data,label),open(dataName+'.data','wb'))
#HCSpace = HC(data, label)
#kernelPara = {'Guassian': [0.1, 1, 10] , 'Polynominal': [(1,1), (1,2), (1,3)]}
#KernelSpace = KernelMapping(HCSpace, kernelPara)
#batch_data, batch_label, data_index = generate_batch(data, label, KernelSpace)


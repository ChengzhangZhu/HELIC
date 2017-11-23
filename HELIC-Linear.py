# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:01:16 2017

@author: Chengzhang Zhu
"""

import tensorflow as tf
import numpy as np
from six.moves import xrange
import matplotlib.pyplot as plt  
import funcs as fc
import pickle
from random import shuffle
from sklearn.cross_validation  import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import shelve




def HC(data, label):
    HCSpace = fc.concatDict([fc.intraAttriEmbed(data), fc.interAttriEmbed(data), fc.VDMEmbed(data, label)])
    return HCSpace
    
def KernelMapping(HCSpace, kernelPara):
    kernelSpace = list()
    for k in kernelPara:
        for para in kernelPara[k]:
            for subspace in HCSpace:
                kernelSpace.append((fc.kernel(HCSpace[subspace], kernelType = k, kernelPara = para), subspace, k, para))
    
    return kernelSpace
       

        
def HML(data, label, dataName, kernelSpace, lam = 1, lr = 1e-2, num_steps = 1000, batch_size = 20):
#    data, label = fc.generate_newInstance(data, label, kernelSpace)
    ###Using Adm to Train the Model
    inputDim = 0
    numOfKernel = len(kernelSpace)
    lam = 1/np.log(numOfKernel)
    for space in kernelSpace:
        inputDim += space[0].shape[0]
#    numOfObj = int((batch_size * batch_size - batch_size)/2)
    numOfObj = batch_size
    ##build the tensorflow graph
    Graph = tf.Graph()
    with Graph.as_default():
        X = tf.placeholder(tf.float32, shape = [None, inputDim]) # the input layer
        Y = tf.placeholder(tf.float32, shape = [None, 1]) # the output ground-truth
        W0 = fc.weight_variable(shape = [inputDim, 1])
        loss = tf.reduce_mean(tf.nn.relu(np.ones([numOfObj,1]) + Y*(tf.matmul(X, tf.abs(W0)) - np.ones([numOfObj,1])))) + lam*tf.reduce_mean(tf.abs(W0))
        solver = (tf.train.AdamOptimizer(learning_rate=lr)
                .minimize(loss))
        W_Out = tf.abs(W0)
        init = tf.global_variables_initializer()
    
    with tf.Session(graph=Graph) as session:
        init.run()
        print("Initialized")
#        data_index = 0
        lossList = list()
        average_loss = 0
        for step in xrange(num_steps):
#            if data_index > data.shape[0] - batch_size or data_index == 0: #shuffle the training data
#                ind_list = [i for i in range(len(data))]
#                shuffle(ind_list)
#                data = data[ind_list]
#                label = label[ind_list]               
#            batch_data, batch_label, data_index = fc.generate_batch(data, label, batch_size, data_index)
            batch_data, batch_label = fc.generate_batch_randomNewInstance(data, label, kernelSpace, batch_size)            
            feed_dict = {X: batch_data, Y: batch_label}
            _, trainLoss, W = session.run([solver,loss, W_Out], feed_dict=feed_dict)
            lossList.append(trainLoss)
            average_loss += trainLoss
            if step % 100 == 0 and step > 0:
                plt.figure()  
                x = np.r_[1:len(lossList)+1]
                plt.plot(x,lossList,"r--",label="loss")  
                plt.xlabel("Iter")  
                plt.ylabel("Loss Value")  
                plt.title("Training Loss")  
                plt.show()
                average_loss /= 100
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
        plt.figure()  
        x = np.r_[1:len(lossList)+1]
        plt.plot(x,lossList,"r--",label="loss")  
        plt.xlabel("Iter")  
        plt.ylabel("Loss Value")  
        plt.ylim(0,6)
        plt.title("Training Loss")  
#        plt.savefig(dataName + '_convergence.eps', format="eps")
    return W

def HMLHC(data, label, dataName = None, kernelPara = {'Guassian': [0.1, 1, 10] , 'Polynominal': [(0,1), (0,2), (0,3)]}, lam = 1, lr = 1e-2, num_steps = 1000, batch_size = 80):
    try:
        HCSpace = pickle.load(open(dataName + 'HCSpace.data', 'rb'))
    except:
        HCSpace = HC(data, label)
        pickle.dump(HCSpace, open(dataName+'HCSpace.data', 'wb'))
    KernelSpace = KernelMapping(HCSpace, kernelPara)
    W = HML(data, label, dataName, KernelSpace, lam, lr, num_steps, batch_size)
    print(len(W))
    embedData = fc.generate_kernelVector(data, KernelSpace, W, reduce = False)
    return embedData, W, KernelSpace
    
dataName = 'census'
#for matlab data
filename = '/data/chezhu/Documents/MATLAB/Coupled Nominal Data for Kernel Learning - v17 non-IID Categorical Data Embedding/Datasets/' + dataName + '.mat'
data, label = fc.readMatlabData(filename)
partition = '/data/chezhu/Documents/MATLAB/Coupled Nominal Data for Kernel Learning - v17 non-IID Categorical Data Embedding/DataPartition/' + dataName + '_Partition.mat'
trnIndex, tstIndex = fc.readMatlabPartition(partition)
###for python data
#s = shelve.open(dataName)
#data = s['X'].astype(np.int32)
#label = s['Y'].astype(np.int32)
#trnIndex, tstIndex = s['Partition']
print('Data Size', len(label))
##split the train and test dataset
#para = np.r_[-5:6]

###for test
#HCSpace = HC(data, label)
#kernelPara = {'Guassian': [2**int(i) for i in para]}
#KernelSpace = KernelMapping(HCSpace, kernelPara)

embed_data, W, KernelSpace = HMLHC(data, label, dataName, lr = 1e-3, batch_size=20, num_steps = 1000, kernelPara = {'Polynominal': [(0,1)]})
##write down results
#pickle.dump((embed_data,label),open(dataName + '_embed.data', 'wb'))
splitNum = 20
index = 0
resKNN = np.zeros([splitNum])

#skf = StratifiedKFold(label[:,0], n_folds = splitNum)
#for train,test in skf:
for p_index in range(20):

    embed_train, embed_test, y_train, y_test = embed_data[trnIndex[p_index,:]-1], embed_data[tstIndex[p_index,:]-1], label[trnIndex[p_index,:]-1], label[tstIndex[p_index,:]-1]

#    embed_train, embed_test, y_train, y_test = embed_data[train], embed_data[test], label[train], label[test]
#        para = np.r_[-5:6]
#        embed_train, W, KernelSpace = HMLHC(X_train, y_train, lr = 1e-2, batch_size=20, num_steps = 200, kernelPara = {'Guassian': [2**int(i) for i in para]})
#        embed_test = fc.generate_kernelVector(X_test, KernelSpace, W)
    neigh = KNeighborsClassifier(n_neighbors = 1)
    neigh.fit(embed_train,y_train[:,0])
#    resKNN[index] = neigh.score(embed_test, y_test[:,0])
    resKNN[index] = f1_score(y_test[:,0], neigh.predict(embed_test), average = 'macro')
    print('the',index,'-th parts KNN accuracy',resKNN[index])
    index += 1
print('the KNN accuracy of HELIC-Linear on data set', dataName,'is:', np.mean(resKNN)*100,'±',np.std(resKNN)*100)
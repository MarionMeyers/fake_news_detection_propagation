#!/usr/bin/env python
# coding: utf-8

# ## GDL Currently working

# In[4]:


import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import os
import torch.nn as nn
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
import torch.nn.functional as F
import random
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x
import torch_geometric.transforms as T
from datetime import datetime

from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# In[7]:


def understandDate(date):
    day, time = date.split('T')
    year, month, day = day.split('-')
    hour,min,sec = time.split(':')
    sec = sec.split('.')[0]

    fullDateString=year+month+day+hour+min+sec
    datetime_object = datetime.strptime(fullDateString, '%Y%m%d%H%M%S')

    return datetime_object

def computeTimeDifference(date1, date2):
    date1 = understandDate(date1)
    date2 = understandDate(date2)
    #The time difference will be computed as a number of seconds, so depending on if we're looking at year, day, month, we
    #will multiply the difference of the 2 by the equivalent number of seconds
    difference=date2-date1
    return difference.total_seconds()


def addTimeBetweenTweets(nodes, edges_features):
    # COMPUTE TIME BETWEEN TWEETS
    # go through all edges, get origin and destination, get the corresponding nodes, get their dates, compute the difference and set it as edge feature
    all_time_diff = []
    for index, edge in edges_features.iterrows():
        # print("edge :"+edge)
        origin = edge['origin']
        destination = edge['destination']
        # print('origin : '+str(origin))

        originNode = nodes[nodes['tweetID'] == origin]
        destinationNode = nodes[nodes['tweetID'] == destination]

        # print('origin node : '+str(originNode))
        # print('destination node : '+str(destinationNode))
        # print("origin node shape : "+str(originNode.shape))
        if (originNode.shape[0] != 0):

            originTime = originNode['date'].values[0]
            destinationTime = destinationNode['date'].values[0]

            # print('origin time : '+str(originTime))
            # print('destination time : '+str(destinationTime))

            diff = computeTimeDifference(originTime, destinationTime)
            # print(diff)
        else:
            diff = None

        # print("time diff : "+str(diff))

        all_time_diff.append(diff)
    # print("edge feature dataframe : " + str(edges_features.head()))

    edges_features['timeDiff'] = all_time_diff

    return nodes, edges_features


# In[8]:


def normalizeNodeFeatures(nodes, edges_features, maxAndMins):
    #maxAndMins = [maxFollowers, minFollowers, maxFollowing, minFollowing, maxFavCount, minFavCount, maxRetCount
                   #  , minRetCount, maxTimeDiff, minTimeDiff]
    maxFollowers=maxAndMins[0]
    minFollowers=maxAndMins[1]
    maxFollowing=maxAndMins[2]
    minFollowing=maxAndMins[3]
    maxFavCount=maxAndMins[4]
    minFavCount=maxAndMins[5]
    maxRetCount=maxAndMins[6]
    minRetCount=maxAndMins[7]
    maxTimeDiff=maxAndMins[8]
    minTimeDiff=maxAndMins[9]
    
    #print('NORM START')
    #print(nodes)
    
    nodes['follower_count'] = nodes['follower_count'].astype(float)
    nodes['following_count'] = nodes['following_count'].astype(float)
    nodes['favCount'] = nodes['favCount'].astype(float)
    nodes['retCount'] = nodes['retCount'].astype(float)
    edges_features['timeDiff'] = edges_features['timeDiff'].astype(float)
    #nodes[["type", 'follower_count', 'following_count', 'favCount','retCount']]
    
    #type=1 for tweets and type=2 for retweet, so we need to create 2 different column features
    tweetType = list()
    retweetType=list()
    
    #print('node matrix shape : '+str(nodes.shape))
    
    def normalize(value, maximum, minimum):
        return (value-minimum)/(maximum-minimum)
    
    nodes['follower_count'] = nodes.apply(lambda row: normalize(row['follower_count'],maxFollowers, minFollowers), axis=1)
    nodes['following_count'] = nodes.apply(lambda row: normalize(row['following_count'],maxFollowing, minFollowing), axis=1)
    nodes['favCount'] = nodes.apply(lambda row: normalize(row['favCount'],maxFavCount, minFavCount), axis=1)
    nodes['retCount'] = nodes.apply(lambda row: normalize(row['retCount'],maxRetCount, minRetCount), axis=1)

    #print('edge feature shape : '+str(edges_features.shape))
    if edges_features.shape[0] != 0:
        edges_features['timeDiff'] = edges_features.apply(lambda row: normalize(row['timeDiff'],maxTimeDiff,minTimeDiff),axis=1)
    
    for index, node in nodes.iterrows():
        if node['type'] == 1:
                tweetType.append(1)
                retweetType.append(0)
        else:
            tweetType.append(0)
            retweetType.append(1)
    nodes['tweetType'] = tweetType
    nodes['retweetType'] = retweetType
    nodes = nodes.drop(columns=["type"])
    
    #print('AFTER NORM')
    #print(nodes)
    #print(edges_features)
    
    return nodes, edges_features
    
        
    


# ## First go-through everything, and compute general measures :
# ### max, min followers, max, min following, max, min fav count, max, min ret count, max min time between tweets

# In[23]:


def initialGraphDataset():
    print('initial graph dataset')
    
                    #FOLLOWER MAX AND MIN
    maxFollowers = 0
    minFollowers = float('inf')
    #FOLLOWING MAX AND MIN
    maxFollowing = 0
    minFollowing = float('inf')
    #FAV COUNT MAX AND MIN
    maxFavCount = 0
    minFavCount = float('inf')
    #RET COUNT MAX AND MIN 
    maxRetCount = 0
    minRetCount = float('inf')
    #TIME BETWEEN TWEETS
    minTimeDiff=float('inf')
    maxTimeDiff=0
    graph_dataset_list = list()
    real_graph_dataset = list()
    fake_graph_dataset = list()
    #graph_features_df = pd.DataFrame(columns=['followerAvg', 'followingAvg', 'graphDensity', 'retweetPercentage', 'avgTimeDiff', 'label'])

    dir = "dir to data set"
    count = 0
    labels = ['real', 'fake']
    #sources = ['politifact','gossipcop']

    realCount = 0
    fakeCount = 0
    #for source in sources: 
     #   dir2 = dir+source+'/'
    for label in labels:

        labelInt = 0
        if label == 'real':
            labelInt = 0
        else:
            labelInt = 1
        print(dir+label+'/')
        
        for news in os.listdir(dir+label):
        #go through news folders in the bucket


            print("news : "+str(news))
            if (labelInt == 1):
                realCount = realCount + 1
            else:
                fakeCount = fakeCount + 1
            if (os.path.exists(dir + label + '/' + str(news) + '/nodes.csv')):
                edges_features = pd.read_csv(dir + label + '/' + str(news) + '/edges.csv', header=None,
                                             names=['origin', 'destination', 'type'])
                nodes = pd.read_csv(dir + label + '/' + str(news) + '/nodes.csv', header=None,
                                   names=['tweetID', 'type', 'date', 'follower_count', 'following_count', 'favCount','retCount', 'userID'])

                nodes, edges_features = addTimeBetweenTweets(nodes, edges_features)
                
                print('nodes: '+str(nodes.shape))
                print('edges features : '+str(edges_features.shape))
                

                #update follower max and min
                if nodes['follower_count'].max() > maxFollowers:
                    maxFollowers = nodes['follower_count'].max()
                    #print('max followers updates to : '+str(maxFollowers))
                if nodes['follower_count'].min() < minFollowers:
                    minFollowers = nodes['follower_count'].min()
                #update following max and min
                if nodes['following_count'].max() > maxFollowing:
                    maxFollowing = nodes['following_count'].max()
                if nodes['following_count'].min() < minFollowing:
                    minFollowing = nodes['following_count'].min()
                #update fav count max and min
                if nodes['favCount'].max() > maxFavCount:
                    maxFavCount = nodes['favCount'].max()
                if nodes['favCount'].min() < minFavCount:
                    minFavCount = nodes['favCount'].min()
                #update ret count max and min
                if nodes['retCount'].max()>maxRetCount:
                    maxRetCount = nodes['retCount'].max()
                if nodes['retCount'].min()<minRetCount:
                    minRetCount = nodes['retCount'].min()
                #update time diff max and min
                if edges_features['timeDiff'].max() > maxTimeDiff:
                    maxTimeDiff = edges_features['timeDiff'].max()
                if edges_features['timeDiff'].min() < minTimeDiff:
                    minTimeDiff = edges_features['timeDiff'].min()


                graph_dataset_list.append([nodes, edges_features, labelInt])
                
    maxAndMins = [maxFollowers, minFollowers, maxFollowing, minFollowing, maxFavCount, minFavCount, maxRetCount
                     , minRetCount, maxTimeDiff, minTimeDiff]

    print("graph dataset list : "+str(len(graph_dataset_list)))
    print('max mins list : '+str(len(maxAndMins)))

    return graph_dataset_list, maxAndMins


#graph_dataset_list, maxAndMins = initialGraphDataset()
#print(len(graph_dataset_list))
#print(maxAndMins)


# In[24]:


def createGraphDataset():
    graph_dataset = list()
    real_graph_dataset = list()
    fake_graph_dataset = list()
    
    graph_dataset_list, maxAndMins = initialGraphDataset()
    #maxAndMins = [maxFollowers, minFollowers, maxFollowing, minFollowing, maxFavCount, minFavCount, maxRetCount
                  #   , minRetCount, maxTimeDiff, minTimeDiff]
    for graph in graph_dataset_list:
        print('new graph in list')
        nodes = graph[0]
        #print('!!!!!!!!!!!!!!!!!!!!nodes type : '+str(type(nodes)))
        edges_features = graph[1]
        labelInt = graph[2]
    
   
     

        """We want to create the edge_index matrix with indices corresponding to the row index of the corresponding
        twees in the nodes features matrix.
        Plan:
        1) Go through all edges
        2) for origin and destination, find the corresponding row indices in the nodes matrix
        3) add a tuple in the edge_index matrix corresponding to row1 and row2"""
        edges_index = np.zeros((2, edges_features.shape[0]))
        
        print('edge features shape : '+str(edges_features.shape))
        print('node features shape : '+str(nodes.shape))
        #for index, row in df.iterrows():
        for index, edge in edges_features.iterrows():
            """print('index : '+str(index))
            print('edge origin '+str(edge.origin))
            print('edge destination : '+str(edge.destination))"""
            tweet1 = edge['origin']
            tweet2 = edge['destination']
            
            """for nodeIndex, node in nodes.iterrows(): 
                print(str(tweet1)+ ' versus '+str(node.tweetID))
                if node.tweetID != tweet1:
                    print('not same')
                else:
                    print('FOUND')"""
                    
            """originNode = nodes[nodes['tweetID'] == tweet1]
            destinationNode = nodes[nodes['tweetID'] ==tweet2]
            
            print('origin node shape : '+str(originNode.shape))
            print('destination node type: '+str(destinationNode.shape))"""
            
            
            tweet1Index = nodes.index[nodes['tweetID'] == tweet1][0]
            tweet2Index = nodes.index[nodes['tweetID'] == tweet2][0]

            edges_index[0][index] = tweet1Index
            edges_index[1][index] = tweet2Index



        nodes, edges_features = normalizeNodeFeatures(nodes, edges_features, maxAndMins)

        nodes = nodes[['follower_count', 'following_count', 'favCount','retCount', 'tweetType','retweetType']]
        #edges_features = edges_features[['type','timeDiff']]
        edges_features = edges_features['timeDiff']
        ##CONVERT EVERYTHING TO PYTORCH TENSORS
        # Convert nodes to x tensor
        nodes = np.array(nodes)
        x = torch.tensor(nodes, dtype=torch.float)

        #Convert edges (only if there are actually some edges in the graph, otherwise it raises an error)
        #print('edge index shape 0 : '+str(edges_index.shape[0]))
        #print('edge index shape 1 : ' + str(edges_index.shape[1]))
        if(edges_index.shape[1] != 0 ):
            edge_index = torch.tensor(edges_index, dtype=torch.long)
            #edge_index = torch.from_numpy(edges_index)
            #print('edge features : '+str(np.array(edges_features)))
            edge_attr = torch.tensor(np.array(edges_features), dtype=torch.float)
            #edge_attr = torch.from_numpy(np.array(edges_features))

            # Create the data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labelInt)
            # add the graph (in pytorch data form) to our dataset

            """print('x shape : '+str(x.shape))
            print('edge index shape : '+str(edges_index.shape))
            print('edge attr shape : '+str(edge_attr.shape))"""

            if(labelInt== 0):
                real_graph_dataset.append(data)
            else:
                fake_graph_dataset.append(data)

            graph_dataset.append(data)
            #print('data appended, graph dataset size : '+str(len(graph_dataset)))

    print('real graph dataset length : '+str(len(real_graph_dataset)))
    print('fake graph dataset length : '+str(len(fake_graph_dataset)))

    print("final graph dataset lenght : "+str(len(graph_dataset)))
    return real_graph_dataset, fake_graph_dataset, graph_dataset


# In[25]:


class Net2(torch.nn.Module):

    #Code taken directly from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py

    def __init__(self):
        #print('init net 2')
        super(Net2, self).__init__()
        numNodesFeatures = len(graph_dataset[0].x[0])
        numClasses = 2
        #self.conv1 = GCNConv(len(d[0].x[0]), 16)
        #self.conv2 = GCNConv(16, 2)
        """lf.conv1 = GCNConv(numNodesFeatures, 3, cached=True)
        self.conv2 = GCNConv(3, numClasses, cached=True)"""

        """self.conv1 = GCNConv(numNodesFeatures, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, numClasses)"""

        #from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py
        self.conv1 = GraphConv(numNodesFeatures, 32)
        #self.conv1 = GCNConv(numNodesFeatures, 32)
        #nn1 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 32))
        #self.conv1 = NNConv(numNodesFeatures, 32, nn1, aggr='mean')

        #nn2 = nn.Sequential(nn.Linear(2, 25), nn.ReLU(), nn.Linear(25, 2048))
        #self.conv2 = NNConv(32, 64, nn2, aggr='mean')
        
        #self.fc1 = torch.nn.Linear(64, 128)
        #self.fc2 = torch.nn.Linear(128, numClasses)
        
        #self.conv1 = NNConv(numNodesFeatures, 32)
        self.pool1 = TopKPooling(32, ratio=0.8)
        self.conv2 = GraphConv(32, 32)
        #self.conv2 = GCNConv(32, 32)
        #self.conv2 = NNConv(32, 32)
        self.pool2 = TopKPooling(32, ratio=0.8)
        """self.conv3 = GraphConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)"""

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 16)
        self.lin3 = torch.nn.Linear(16, numClasses)

    def forward(self, data):
        #print('net 2 forward method called with parameters ' +str(data))

        #again, all of it from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x,edge_index, edge_attr))
        #x=F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _,_ = self.pool1(x, edge_index, edge_attr, batch)
        #x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch)
        #x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        #x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print('batch size x1 :' + str(batch.shape))
        #print('x1 shape : ' + str(x1.shape))

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _,_ = self.pool2(x, edge_index, edge_attr, batch)
        #x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        """
        #print('batch size x2 :' + str(batch.shape))
        #print('x2 shape : ' + str(x2.shape))

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)""""""
        #print('batch size x3 :' + str(batch.shape))
        #print('x3 shape : ' + str(x3.shape))

        x = x1 + x2 + x3"""
        x=x1 +x2
        
        x = F.relu(self.lin1(x))
        #print('x after lin1 : ' + str(x.shape))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        #print('x after lin2 : ' + str(x.shape))
        x = F.log_softmax(self.lin3(x), dim=-1)
        #print('x after softmax : ' + str(x.shape))
        

        return x




#CREATE DATASET
real_graph_dataset, fake_graph_dataset, graph_dataset = createGraphDataset()


random.shuffle(graph_dataset)
random.shuffle(real_graph_dataset)
random.shuffle(fake_graph_dataset)
"""MAKE THE DATASET BALANCED between true and fake news
take the smallest size out of the 2 sets"""
finalLength=0
if(len(real_graph_dataset) < len(fake_graph_dataset)):
    finalLength = len(real_graph_dataset)
else:
    finalLength = len(fake_graph_dataset)
print(finalLength)

#real_graph_dataset_new = real_graph_dataset[0:finalLength]
#fake_graph_dataset_new = fake_graph_dataset[0:finalLength]
real_graph_dataset_new =random.sample(real_graph_dataset, finalLength)
fake_graph_dataset_new =random.sample(fake_graph_dataset, finalLength)

graph_dataset_new = real_graph_dataset_new + fake_graph_dataset_new

random.shuffle(graph_dataset_new)

print(len(graph_dataset_new))


# ## Cross validation implementation
#FOR LARGE DATASET
random.shuffle(graph_dataset_new)
gf = graph_dataset_new[0:620]
k=10
folds = list()
fold_size = 62
i=0
start=0
end=fold_size
while i<k:
    folds.append(gf[start:end])
    start = start +fold_size
    end = end+fold_size
    i = i+1

# In[13]:


#We consider fake news to be the positive label: y=1
def computeF1Score(modelOutput, correctLabels):
    #print(type(modelOutput))
    positiveOutputs = modelOutput.count(1)
    positiveLabels = correctLabels.count(1)
    
    correctPositive = 0
    
    for i in range(0,len(modelOutput)):
        #if it should have been positive
        if correctLabels[i] ==1 and modelOutput[i] == 1:
            correctPositive+=1
    #print('positive model outputs : '+str(positiveOutputs))
    #print('positive labels : '+str(positiveLabels))
    #print('correct positive : '+str(correctPositive))
            
    #P = PRECISION
    p=0
    if positiveOutputs != 0 : 
        p = correctPositive/positiveOutputs
    
    #R = RECALL
    r=0
    if positiveLabels != 0 :
        r = correctPositive/positiveLabels
    
    f1Score =0
    if (p+r) != 0 :
        f1Score = 2 * ((p*r)/(p+r))
    return p,r,f1Score
    


# In[74]:


def computeRates(modelOutputs, correctLabels):
    #where 1 (fake news) == positive and 0 (true news ) == negative
    tp=0
    fn=0
    fp=0
    tn=0
    
    for i in range(0,len(modelOutputs)):
        #TRUE POS
        if correctLabels[i] == 1 and modelOutputs[i]==1:
            tp+=1
        #FALSE POS
        elif correctLabels[i] == 0 and modelOutputs[i] ==1:
            fp+=1
         
        #FALSE NEG  
        elif correctLabels[i] == 1 and modelOutputs[i] == 0:
            fn+=1
        
        #TRUE NEG
        else:
            tn+=1
    tpRate=0
    fpRate=0
    if (tp+fn)!=0:
        tpRate = tp/(tp+fn)
    else:
        print('tp rate 0 because no pos output')
    
    if (fp+tn)!=0:
        fpRate = fp/(fp+tn)
    else:
        print('fp rate 0 because no neg output')
    
    
    return tp, tn, fp, fn, tpRate, fpRate
            
            
        


# In[ ]:


#model_to_try = Net2()
epochs = 400
def train(epoch):
    model.train()

    #all from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py
    loss_all = 0
    for data in train_loader:
        #print("new batch")
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        #print('loss all: '+str(loss_all))
        optimizer.step()
    return loss_all / len(trainingFolds)
    #return loss_all/len(trainSet2)

def test():
    model.eval()
    correctTrain=0
    correctTest=0
    modelOutputTrain=list()
    correctLabelsTrain = list()
    modelOutputTest=list()
    correctLabelsTest = list()
    
    print('train loader shape : '+str(len(train_loader)))
    print('test loader shape : '+str(len(test_loader)))
    
    
    #Get the scores on the training set
    for data in train_loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        #print('pred : '+str(pred))
        #print('actual y : '+str(data.y))
        modelOutputTrain.append(pred)
        correctLabelsTrain.append(data.y)
        correctTrain+= pred.eq(data.y).sum().item()
    
    pTrain,rTrain,f1Train = computeF1Score(modelOutputTrain, correctLabelsTrain)
    tpTrain, tnTrain, fpTrain, fnTrain, tpRateTrain, fpRateTain = computeRates(modelOutputTrain, correctLabelsTrain)
    print('correct trains : '+str(correctTrain))
    print('training accuracy : '+str(correctTrain/len(trainingFolds)))
    
    print('TRAINING CONF matrix')
    #print('tpTrain : '+str(tpTrain))
    #print('tnTrain : '+str(tnTrain))
    #print('fpTrain : '+str(fpTrain))
    #print('fnTrain : '+str(fnTrain))
    #get the scores on the test set
    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        #print('pred : '+str(pred))
        #print('actual y : '+str(data.y))
        modelOutputTest.append(pred)
        correctLabelsTest.append(data.y)
        correctTest+= pred.eq(data.y).sum().item()
    
    pTest,rTest,f1Test = computeF1Score(modelOutputTest, correctLabelsTest)
    tpTest,tnTest, fpTest,fnTest, tpRateTest, fpRateTest = computeRates(modelOutputTest, correctLabelsTest)
    print('correct tests : '+str(correctTest))
    print('testing accuracy : '+str(correctTest/len(testFold)))
    print('TESTING CONF matrix')
    #print('tpTest : '+str(tpTest))
    #print('tnTest : '+str(tnTest))
    #print('fpTest : '+str(fpTest))
    #print('fnTest : '+str(fnTest))
    #print('precision : '+str(p))
    #print('recall : '+str(r))
    #print('f1 score : '+str(f1))
    #print('true positive rate : '+str(tpRate))
    #print('false positive rate : '+str(fpRate))
    return [correctTrain/len(trainingFolds), pTrain, rTrain, f1Train, tpRateTrain, fpRateTain, tpTrain, tnTrain, fpTrain, fnTrain],[correctTest/len(testFold), pTest, rTest, f1Test, tpRateTest, fpRateTest,tpTest, tnTest, fpTest, fnTest]
    #return correct/len(testSet2)

accuracyScores=list()
foldsScoreEvolutionsTrain=list()
foldsScoreEvolutionsTest=list()

for i in range(0,k):
    
    #create new score evolution dataframe
    scoresEvolTrain = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'tpRate', 'fpRate'])
    scoresEvolTest = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'tpRate', 'fpRate'])
    
    """RE-INITIALIZE THE MODEL PARAMETERS TO START FRESH ON A NEW FOLD"""
    transform = T.Cartesian(cat=False)

    device = torch.device('cpu')
    model = Net2().to(device)
    #model = model_to_try.to(device)
    print('model type after to  : '+str(type(model)))
    #print('data type after to : '+str(type(data)))
    print('model parameters : '+str(model.parameters))
    #data = graph_dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    #optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4),dict(params=model.non_reg_params, weight_decay=0)], lr=0.001)
    
    
    
    foldScoresTrain=list()
    foldScoresTest=list()
    print('NEW FOLD : '+str(i))
    trainingFolds = list()
    testFold = list()
    
    for j in range(0,k):
        if j==i:
        	testFold = folds[i]
        else:
        	trainingFolds.extend(folds[j])
    print('test fold')
    for test_fold in testFold:
        print(test_fold.y)
    print('training folds')
    for training_fold in trainingFolds:
        print(training_fold.y)

    train_loader= DataLoader(trainingFolds, batch_size=1, shuffle=True)
    test_loader = DataLoader(testFold, batch_size=1, shuffle=True)
    
    print('train loader size : '+str(len(train_loader)))
    print('test loader size : '+str(len(test_loader)))
    
    
    scoresTrain=list()
    scoresTest=list()
    for epoch in range(0,epochs):
        train(epoch)
        scoresTrain, scoresTest = test()
        optimizer.zero_grad()
        print('Epoch: {:02d}, Test: {:.4f}, p:{:.4F}, r:{:.4F},f1:{:.4F}, tpr:{:.4F}, fpr:{:.4F}'.format(epoch, scoresTest[0], scoresTest[1],
                                                                                                        scoresTest[2], scoresTest[3], scoresTest[4], scoresTest[5]))
        print('Epoch: {:02d}, Train: {:.4f}, p:{:.4F}, r:{:.4F},f1:{:.4F}, tpr:{:.4F}, fpr:{:.4F}'.format(epoch, scoresTrain[0], scoresTrain[1],
                                                                                                        scoresTrain[2], scoresTrain[3], scoresTrain[4], scoresTrain[5]))
       
        foldScoresTrain.append(scoresTrain)
        foldScoresTest.append(scoresTest)
        #scoresEvol.append(scores)
        scoresEvolTrain = scoresEvolTrain.append({'accuracy':scoresTrain[0], 'precision':scoresTrain[1], 'recall':scoresTrain[2], 'f1':scoresTrain[3],'tpRate':scoresTrain[4],'fpRate':scoresTrain[5], 'tpTrain':scoresTrain[6], 'tnTrain':scoresTrain[7], 'fpTrain':scoresTrain[8], 'fnTrain':scoresTrain[9]}, ignore_index=True)
        scoresEvolTest = scoresEvolTest.append({'accuracy':scoresTest[0], 'precision':scoresTest[1], 'recall':scoresTest[2], 'f1':scoresTest[3],'tpRate':scoresTest[4],'fpRate':scoresTest[5],'tpTest':scoresTest[6], 'tnTest':scoresTest[7], 'fpTest':scoresTest[8], 'fnTest':scoresTest[9]}, ignore_index=True)
        #print('scores evol : '+str(scoresEvol.shape))
        
    #accuracyScores.append(foldScores[epochs-1])
    #these are lists of dataframes (one dataframe per fold)
    #foldsScoreEvolutionsTrain.append(scoresEvolTrain)
    #foldsScoreEvolutionsTest.append(scoresEvolTest)
    torch.save(model, 'model_save_fold_'+str(i))
    scoresEvolTrain.to_csv('no_time_GraphConv_gdl_scores' + '_Net2_' + str(epochs) +'_fold'+str(i)+ '_training_scores.csv')
    scoresEvolTest.to_csv('no_time_Graph_Conv_gdl_scores' + '_Net2_'+ str(epochs) +'_fold'+str(i)+ '_testing_scores.csv')

#foldsScoreEvolutionsTrain.to_csv('gdl_scores'+str(model_to_try)[0:4]+'_'+str(epochs)+'_training_scores.csv')
#foldsScoreEvolutionsTest.to_csv('gdl_scores'+str(model_to_try)[0:4]+_+str(epochs)+'_testing_scores.csv')


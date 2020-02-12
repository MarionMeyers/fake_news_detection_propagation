#!/usr/bin/env python
# coding: utf-8

# # Building an SVM in simply extracted graph features
# ## Features extracted are : average user follower, average user followings, density of the graph 

# In[104]:


import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.utils
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# ## List of methods used to extract different features from the edges and nodes dataframes

# In[37]:


def computeFollowerAvg(nodesDf):
    return nodesDf['follower_count'].mean()

def computeFollowingAvg(nodesDf):
    return nodesDf['following_count'].mean()

def computeGraphDensity(nodesDf, edgesDf):
    numbNodes = nodesDf.shape[0]
    #print("number of nodes : "+str(numbNodes))
    numbEdges = edgesDf.shape[0]
    density = 1
    if(numbNodes != 0 and numbNodes!=1):
        density = (2*numbEdges)/(numbNodes*(numbNodes-1))
    return density

def computeRetweetPercentage(nodesDf):
    tweetRows = nodesDf['type'] == 1
    tweets = nodesDf[tweetRows]
    
    retweetRows = nodesDf['type'] == 2
    retweets = nodesDf[retweetRows]
    #print("number of tweets : "+str(len(tweets)))
    #print("number of retweets : "+str(len(retweets)))
    
    return len(retweets)/(len(tweets)+len(retweets))

def computeAvgTimeDiff(edgesDf):
    return edgesDf['timeDiff'].mean()
    

def computeNumbTweets(nodesDf):
    tweets = nodesDf[nodesDf['type'] ==1]
    #print('num Tweets'+str(tweets.shape[0]))
    return tweets.shape[0]

def computeNumbRetweets(nodesDf):
    retweets = nodesDf[nodesDf['type'] ==2 ]
    #print('num retweets : '+str(retweets.shape[0]))
    return retweets.shape[0]

def computeTimeBetweenFirstLastTweet(nodesDf):
    #take the first tweet of the dataframe (if the rest of the code is correct then the df is sorted already)
    #print('nodes matrix size : '+str(nodesDf.shape))
    
    firstTweet = nodesDf.iloc[0]
    lastTweet = nodesDf.iloc[nodesDf.shape[0] - 1]
    
    date1 = understandDate(firstTweet['date'])
    date2 = understandDate(lastTweet['date'])
    
    return (date2-date1).total_seconds()

def computeAvgFavorite(nodesDf):
    fav = nodesDf['favorite_count']
    return fav.mean()

def computeAvgRetCount(nodesDf):
    #print('calc ret count')
    tweets = nodesDf[nodesDf['type'] == 1]
    retCount = tweets['retweet_count']
    #retCountSum = retCount.sum()
    #print('ret count sum : '+str(retCountSum))
    
    #print(retCount.mean())
    
    return retCount.mean()
def computeEdgesToNodes(nodesDf, edgesDf):
    #print('edges to nodes')
    #print('edges length : '+str(edgesDf.shape[0]))
    #print('nodes length : '+str(nodesDf.shape[0]))
    if nodesDf.shape[0] !=0:
        #print(edgesDf.shape[0]/nodesDf.shape[0])
        return edgesDf.shape[0]/nodesDf.shape[0]
    else:
        return 0
    
def computeNodesToEdges(nodesDf, edgesDf):
    if edgesDf.shape[0]!=0:
        return nodesDf.shape[0]/edgesDf.shape[0]
    else:
        return 0


# In[38]:


def computeUsersTouched10hours(nodesDf):
    usersTouched = list()
    """structure:
    go through all tweets in a time order
    of the time difference between the first and this tweet is smaller than 1 day, then store the tweet
    
    Now we have a list of all tweets/retweets that happened withing one day of the first one
    
    collect the users involved and count the number of different ones in the list"""
    #print('nodes df shape : '+str(nodesDf.shape))
    firstDate= nodesDf.iloc[0]['date']
    #print('first tweet type : '+str(type(firstTweet)))
    
    #tweetsWithinDay=list()
    for index, node in nodesDf.iterrows():
        timeDiff = computeTimeDifference(firstDate, node['date'])
        
        #there are 86 400 seconds in one day yes
        if node['userID'] not in usersTouched and timeDiff <= 36000: 
            #tweetsWithinDay.append(node)
            usersTouched.append(node['userID'])
    
    return len(usersTouched)


# In[39]:


def timeBefore100(nodesDf):
    firstDate= nodesDf.iloc[0]['date']
    lastDate = nodesDf.iloc[1000]['date']
    timeDiff = computeTimeDifference(firstDate, lastDate)
    return timeDiff


# In[40]:


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


# In[41]:


def addTimeBetweenTweets(nodes, edges_features):
    
    """Because we only have retweet link atm, the mean time we will compute can be seen as the average time of retweets 
    after a tweet"""
    
    #COMPUTE TIME BETWEEN TWEETS
    #go through all edges, get origin and destination, get the corresponding nodes, get their dates, compute the difference and set it as edge feature
    all_time_diff = []
    for index, edge in edges_features.iterrows():
        #print("edge :"+edge)
        origin = edge['origin']
        destination = edge['destination']
        #print('origin : '+str(origin))

        originNode = nodes[nodes['tweetID']==origin]
        destinationNode= nodes[nodes['tweetID']==destination]

        #print('origin node : '+str(originNode))
        #print('destination node : '+str(destinationNode))
        #print("origin node shape : "+str(originNode.shape))
        if(originNode.shape[0] != 0):

            originTime = originNode['date'].values[0]
            destinationTime = destinationNode['date'].values[0]

            #print('origin time : '+str(originTime))
            #print('destination time : '+str(destinationTime))

            diff = computeTimeDifference(originTime, destinationTime)
            #print(diff)
        else:
            diff = None
        
        #print("time diff : "+str(diff))
        
        all_time_diff.append(diff)
    #print("edge feature dataframe : " + str(edges_features.head()))

    edges_features['timeDiff'] = all_time_diff
    
    return nodes, edges_features


# In[36]:


def computePercTweets1hour(nodes):
    
    startDate = nodes.iloc[0]['date']
    before1hour = 0
    for index, node in nodes.iterrows():
        timeDiff = computeTimeDifference(startDate, node['date'])
        if timeDiff < 3600:
            before1hour+=1
            
    perc = 0 
    if len(nodes)>0:
        perc = before1hour/len(nodes)
    return perc


# In[54]:


import math
def newsEvolutionOverTime(nodes):
    nodes = nodes.sort_values(by=['date'])
    #nodes = nodes.sort_values(by=['date'])
    #put the start time as 'hour 0' and the max end time of all graphs as the last possible hour
    #we will probably have to cut graphs because the x axis will be very long
    #print('nodes shape : '+str(nodes.shape))
    startDate = nodes.iloc[0]['date']
    print('start date : '+str(startDate))
    #print(nodes.shape[0])
    endDate = nodes.iloc[nodes.shape[0]-1]['date']
    print('end date : '+str(endDate))
    #print('start date :'+str(startDate))
    #print('end date : '+str(endDate))
    max_time_diff = computeTimeDifference(startDate, endDate)
    hours = np.zeros(math.ceil((max_time_diff)/3600))
    
    if len(hours) ==0:
        return [1]
    #print('hours shape : '+str(len(hours)))
    hour_diff=0
    hour_index = 1
    #are they already ordered by date???
    #this code is only working if the nodes are ordered by date
    for index, node in nodes.iterrows(): 
        #print('node date : '+str(node['date']))
        time_diff = computeTimeDifference(startDate, node['date'])
        node_hour = int(time_diff/3600)
        hours[node_hour] = hours[node_hour]+1
        #print('node index : '+str(index))
    
    total_nodes = nodes.shape[0]
    #print('total number of nodes : '+str(total_nodes))
    hours = hours/total_nodes
    #print('hours in percentages : '+str(hours))
    #print('hours sum : '+str(sum(hours)))
    
    
    #now i want the last one to add up to a 100%
    for index in range(0, len(hours)):
        if index!=0:
            hours[index] = hours[index] + hours[index-1]
        
    return hours

graph_dataset=list()
hour_evolution_real = list()
hour_evolution_fake = list()


graph_features_df = pd.DataFrame(columns=['numNodes','numEdges','followerAvg','followingAvg','graphDensity','retweetPercentage',
                                          'avgTimeDiff','numTweets','numRetweets', 'avgFav', 'avgRetCount',
                                          'time_first_last', 'edgesToNodes','nodesToEdges', 'usersTouched10hours',
                                          'percTweets1hour','label'])

dir = "insert the directory to the politifact folder"
count = 0 
labels =['real','fake']
sources = ['politifact','gossipcop']

realCount=0
fakeCount=0
for label in labels:

    labelInt=0
    if label == 'real':
        labelInt=1
    else:
        labelInt=2
    print(dir+label+'/')
    for news in os.listdir(dir+label+'/'):
        if(labelInt==1):
            realCount = realCount+1
        else:
            fakeCount = fakeCount+1
        #print("count : "+str(count))
        #print(dir+label+'/'+str(news)+'/nodes.csv')
        if(os.path.exists(dir+label+'/'+str(news)+'/nodes.csv')):

            #print(news + "nodes and edges found")
            edges_features = pd.read_csv(dir+label+'/'+str(news)+'/edges.csv', header=None, names=['origin', 'destination', 'type'])
            nodes = pd.read_csv(dir+label+'/'+str(news)+'/nodes.csv', header=None, names=['tweetID', 'type', 'date', 
                                                                                          'follower_count','following_count',
                                                                                         'favorite_count', 'retweet_count', 'userID'])

            #print("nodes types : "+str(type(nodes)))
            numNodes = len(nodes)
            numEdges = len(edges_features)
            
            #print('NUM NODES : '+str(numNodes))
            #print('NUM EDGES : '+str(numEdges))

            followerAvg = computeFollowerAvg(nodes)
            #print("average follower count : "+str(followerAvg))

            followingAvg = computeFollowingAvg(nodes)
            #print("average following count : "+str(followingAvg))

            if followerAvg > 75000:
                print(str(label)+' news '+str(news)+' has followerAvg : '+str(followerAvg))

            if followingAvg > 10000:
                print('news '+str(news)+' has followingAvg : '+str(followingAvg))

            
            
            
            #for now, if the number of nodes is 1 or 0 , then the density of the graph will be 1
            graphDensity = computeGraphDensity(nodes, edges_features)
            #print("graph density : "+str(graphDensity))

            retweetPercentage = computeRetweetPercentage(nodes)

            avgTimeDiff = computeAvgTimeDiff(edges_features)

            #print("average time diff : "+str(avgTimeDiff))

            #graph_info = pd.Series([followerAvg,followingAvg,graphDensity, avgTimeDiff])

            numTweets = computeNumbTweets(nodes)
            numRetweets = computeNumbRetweets(nodes)

            
            if numTweets > 2000:
                print(str(label)+' news '+str(news)+' has num Tweets : '+str(numTweets))
                
            time_first_last = computeTimeBetweenFirstLastTweet(nodes)
            
            if time_first_last > 210000000:
                print(str(label)+' news '+str(news)+' has lifetime : '+str(time_first_last))

            avgFav = computeAvgFavorite(nodes)

            avgRetCount = computeAvgRetCount(nodes)

            edgesToNodes = computeEdgesToNodes(nodes, edges_features)
            nodesToEdges = computeNodesToEdges(nodes, edges_features)

            usersTouched10hours = computeUsersTouched10hours(nodes)

            percTweets1hour = computePercTweets1hour(nodes)
            #print(percTweets1hour)
            #timeBef1000 = timeBefore1000(nodes)
            
            if label == 'real':
                hour_evolutions_real = hour_evolution_real.append(newsEvolutionOverTime(nodes))
            else:
                hour_evolutions_fake = hour_evolution_fake.append(newsEvolutionOverTime(nodes))

            graph_features_df = graph_features_df.append({'numNodes':numNodes,
                                                          'numEdges':numEdges,
                                                          'followerAvg':followerAvg, 
                                                          'followingAvg':followingAvg,
                                                          'graphDensity':graphDensity, 
                                                          'retweetPercentage':retweetPercentage,
                                                          'avgTimeDiff':avgTimeDiff, 
                                                          'numTweets':numTweets, 
                                                          'numRetweets':numRetweets, 
                                                          'time_first_last':time_first_last,
                                                          'avgFav':avgFav,
                                                          'avgRetCount':avgRetCount,
                                                          'edgesToNodes':edgesToNodes,
                                                          'nodesToEdges':nodesToEdges,
                                                          'usersTouched10hours':usersTouched10hours,
                                                          'percTweets1hour' :percTweets1hour,
                                                          'label':labelInt

                                                         }, ignore_index=True )



# ## Visualizing hour evolutions
#1. get the max number of hours, and make it the x label
#2. go through each evolution, and extend them with 1 starting from their end position to the max end position
#3. sum all lists together, and divide through number of evolutions
def visualize_hour_evolutions(hour_evolution_real, hour_evolution_fake):
    #get the max size
    max_hour=0

    for evol in hour_evolution_real:
        if len(evol)>max_hour:
            max_hour = len(evol)
    for evol in hour_evolution_fake:
        if len(evol)>max_hour:
            max_hour = len(evol)
    hour_evol_real = pd.DataFrame(columns=range(0,max_hour))
    hour_evol_fake = pd.DataFrame(columns=range(0,max_hour))

    for i, evol in enumerate(hour_evolution_real):
        new_list = np.ones(max_hour)
        for index, perc in enumerate(evol):
            new_list[index] = perc
        series = pd.Series(new_list)
        hour_evol_real = hour_evol_real.append(series, ignore_index=True)

    for i, evol in enumerate(hour_evolution_fake):
        new_list = np.ones(max_hour)
        for index, perc in enumerate(evol):
            new_list[index] = perc
        series = pd.Series(new_list)
        hour_evol_fake = hour_evol_fake.append(series, ignore_index=True)


    old_hours = range(1,hour_evol_real.shape[1]+1)
    hours=list()
    for i,hour in enumerate(old_hours):
        hour= hour+1
        hours.append(hour)
    real_sums = hour_evol_real.sum()
    real_percentages = real_sums/hour_evol_real.shape[0]
    plt.plot(hours, real_percentages, label='real news')

    fake_sums = hour_evol_fake.sum()
    fake_percentages = fake_sums/hour_evol_fake.shape[0]
    plt.plot(hours, fake_percentages, label='fake news')
    plt.xlabel("number of hours")
    plt.ylabel('percentage of posts')
    plt.legend(loc=2)
    plt.show()

    old_early_hours = range(1,8001)
    early_hours=list()
    for i,hour in enumerate(old_early_hours):
        hour= hour+1
        early_hours.append(hour)

    early_hours_real = real_percentages[0:8000]
    early_hours_fake = fake_percentages[0:8000]
    plt.plot(early_hours, early_hours_real, label='real news')
    plt.plot(early_hours, early_hours_fake, label='fake news')
    plt.xlabel("number of hours")
    plt.ylabel('percentage of posts')
    plt.legend(loc=2)
    plt.show()

    old_early_hours = range(1,301)
    early_hours=list()
    for i,hour in enumerate(old_early_hours):
        hour= hour+1
        early_hours.append(hour)

    early_hours_real = real_percentages[0:300]
    early_hours_fake = fake_percentages[0:300]
    plt.plot(early_hours, early_hours_real, label='real news')
    plt.plot(early_hours, early_hours_fake, label='fake news')
    plt.xlabel("number of hours")
    plt.ylabel('percentage of posts')
    plt.legend(loc=2)
    plt.show()

    early_hours=range(0,24)
    early_hours_real = real_percentages[0:24]
    early_hours_fake = fake_percentages[0:24]
    plt.plot(early_hours, early_hours_real, label='real news')
    plt.plot(early_hours, early_hours_fake, label='fake news')
    plt.xlabel("number of hours")
    plt.ylabel('percentage of posts')
    plt.legend(loc=2)
    plt.show()


def tweet_sums(graph_features_df):
    #Let's sum up all the tweets and retweets
    sumTweets = graph_features_df['numTweets'].sum()
    #let's sum up all the retweets
    sumRetweets = graph_features_df['numRetweets'].sum()

    print('tweet sum : '+str(sumTweets))
    print('retweets sum : '+str(sumRetweets))



colnull =graph_features_df[graph_features_df['avgTimeDiff'].isna()]
colnull.head()

#DROP THE ROWS WITH A 'NAN' in it 
graph_features_df = graph_features_df.dropna(0)
print(graph_features_df.shape)




# ## Some preliminary analysis on the dataset and features

# Let us do some analysis on the average time difference features, comparing between true and false graphs

# In[51]:


realGraphs = graph_features_df[graph_features_df['label'] == 1.0]
#print(realGraphs.head())
fakeGraphs = graph_features_df[graph_features_df['label'] == 2.0]
print(realGraphs.shape)
print(fakeGraphs.shape)




"""AVERAGE TIME DIFFERENCE"""
avgTimeDiffReal = realGraphs['avgTimeDiff'].mean()
avgTimeDiffFalse = fakeGraphs['avgTimeDiff'].mean()

print("avg false time diff : "+str(avgTimeDiffFalse))
print("avg real tume diff : "+str(avgTimeDiffReal))


"""AVERAGE NUMBER OF FOLLOWERS"""
avgFollowersReal = realGraphs['followerAvg'].mean()
avgFollowersFake = fakeGraphs['followerAvg'].mean()
print('avg follower real '+str(avgFollowersReal))
print('avg follower fake :'+str(avgFollowersFake))

"""AVERAGE NUMBER OF FOLLOWERS"""
avgFollowingReal = realGraphs['followingAvg'].mean()
avgFollowingFake = fakeGraphs['followingAvg'].mean()
print('avg following real : '+str(avgFollowingReal))
print('avg following fake :'+str(avgFollowingFake))

"""AVERAGE GRAPH DENSITY"""
avgGraphDensityReal = realGraphs['graphDensity'].mean()
avgGraphDensityFake = fakeGraphs['graphDensity'].mean()
print('avg graph density real : '+str(avgGraphDensityReal))
print('avg graph density fake : '+str(avgGraphDensityFake))

"""AVERAGE RETWEET PERCENTAGE"""
avgRetweetPercReal = realGraphs['retweetPercentage'].mean()
avgRetweetPercFake = fakeGraphs['retweetPercentage'].mean()
print('avg retweet percentage real : '+str(avgRetweetPercReal))
print('avg retweet percentage fake : '+str(avgRetweetPercFake))

"""AVERAGE NUMBER OF TWEETS"""
avgNumTweetsReal = realGraphs['numTweets'].mean()
avgNumTweetsFake = fakeGraphs['numTweets'].mean()
print('average numb tweets real : '+str(avgNumTweetsReal))
print('average num tweets fake : '+str(avgNumTweetsFake))

"""AVERAGE NUMBER OF RETWEETS"""
avgNumRetweetsReal = realGraphs['numRetweets'].mean()
avgNumRetweetsFake = fakeGraphs['numRetweets'].mean()
print('avg numb retweets real : '+str(avgNumRetweetsReal))
print('avg numb retweets fake : '+str(avgNumRetweetsFake))

"""AVERAGE LAST-FIRST TWEET TIME"""
avgLastFirstTimeReal = realGraphs['time_first_last'].mean()
avgLastFirstTimeFake= fakeGraphs['time_first_last'].mean()
print('avg first_last time real : '+str(avgLastFirstTimeReal))
print('avg first_last time fake : '+str(avgLastFirstTimeFake))

"""AVERAGE FAV COUNT"""
avgFavCountReal  = realGraphs['avgFav'].mean()
avgFavCountFake = fakeGraphs['avgFav'].mean()
print('avg fav count real : '+str(avgFavCountReal))
print('avg fav count fake : '+str(avgFavCountFake))

"""AVERAGE RETWEET COUNT"""
avgRetCountReal = realGraphs['avgRetCount'].mean()
avgRetCountFake = fakeGraphs['avgRetCount'].mean()
print('avg ret count real : '+str(avgRetCountReal))
print('avg ret count fake : '+str(avgRetCountFake))

"""AVERAGE Edges to Nodes COUNT"""
avgEtNReal = realGraphs['edgesToNodes'].mean()
avgEtNFake = fakeGraphs['edgesToNodes'].mean()
print('avg edges to nodes real : '+str(avgEtNReal))
print('avg edges to nodes fake : '+str(avgEtNFake))

"""AVERAGE Nodes to Edges COUNT"""
avgNtEReal = realGraphs['nodesToEdges'].mean()
avgNtEFake = fakeGraphs['nodesToEdges'].mean()
print('avg nodes to edges real : '+str(avgNtEReal))
print('avg nodes to edges fake : '+str(avgNtEFake))

"""AVERAGE Users touched in 1 day  COUNT"""
avg10hoursReal = realGraphs['usersTouched10hours'].mean()
avg10hoursFake = fakeGraphs['usersTouched10hours'].mean()
print('avg users 10 hours real : '+str(avg10hoursReal))
print('avg users 10 hours fake : '+str(avg10hoursFake))

"""AVERAGE perc tweets 1 hour  COUNT"""
avgPercTweets1hourReal = realGraphs['percTweets1hour'].mean()
avgPercTweets1hourFake = fakeGraphs['percTweets1hour'].mean()
print('avg perc tweets 1 hour : '+str(avgPercTweets1hourReal))
print('avg perc tweets 1 hour : '+str(avgPercTweets1hourFake))


# ## T-Test on some of the features

# In[8]:


from scipy import stats
import math
def computeTStat(s1, s2, mean1, mean2, n1, n2):
    
    print('s1 : '+str(s1))
    print('s2 : '+str(s2))
    print('mean1 : '+str(mean1))
    print('mean2 : '+str(mean2))
    
    s1n1 = math.pow(s1,2)/n1
    s2n2 = math.pow(s2,2)/n2
    print('s1n1 : '+str(s1n1))
    print('s2n2 : '+str(s2n2))
    SE = math.sqrt(s1n1+s2n2)
    print('se : '+str(SE))
    
    
    
    degFreedom = (math.pow(s1n1+s2n2, 2)) / ((math.pow(s1n1,2)/(n1-1))+(math.pow(s2n2,2)/(n2-1)))
    print('degrees of freedom : '+str(degFreedom))
    #test = ((math.pow(math.pow(s1,2)/n1,2)/(n1-1)) +(math.pow(math.pow(s2,2)/n2,2)/(n2-1))
    #degFreedom = ((    math.pow( math.pow(s1,2)/ n1 + math.pow(s2,2)/n2 ,2)  ) / ((math.pow(math.pow(s1,2)/n1,2)/(n1-1)) +(math.pow(math.pow(s2,2)/n2,2)/(n2-1)))
    t = (mean1 - mean2)/SE
    return degFreedom, t


# In[ ]:


feature_stats = pd.DataFrame(columns=['avg_real', 'avg_fake', 'std_real', 'std_fake', 't', 'p_value'])
print(feature_stats.columns)

for col in graph_features_df.columns:
    if col =='label':
        break
    print("feature : "+str(col))
    avg_real= realGraphs[col].mean()
    avg_fake = fakeGraphs[col].mean()
    std_real= realGraphs[col].std()
    std_fake = fakeGraphs[col].std()
    
    #data = pd.Series(avg_real, avg_fake, std_real, std_fake, t2[0], t2[1])
    #print(avg_real)

    t = computeTStat(std_real, std_fake, avg_real, avg_fake, len(realGraphs), len(fakeGraphs))
    t2 = stats.ttest_ind(realGraphs[col],fakeGraphs[col], equal_var = False)
    print(t)
    print("second results : "+str(t2))
    print(type(t2))
    new_row = {'avg_real':"{:.4f}".format(avg_real), 'avg_fake':"{:.4f}".format(avg_fake), 'std_real':"{:.4f}".format(std_real), 'std_fake':"{:.4f}".format(std_fake),'t': "{:.4f}".format(t2.statistic),'p_value': "{:.4f}".format(t2.pvalue)}
    feature_stats = feature_stats.append(new_row, ignore_index=True)

print(feature_stats.columns)
feature_stats.to_latex(index=False)


def plot_boxplots(graph_features_df):
    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'percTweets1hour', data = graph_features_df, showfliers=False)
    boxplot.set_title("Percentage of Posts Within 1st Hour (Without Outliers)", fontsize=15)
    boxplot.set( ylabel='Perc Posts 1st hour')
    boxplot.set_xticklabels(['true', 'false'])

    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'time_first_last', data = graph_features_df,  showfliers=True)
    boxplot.set_title("News Lifetime (With Outliers)", fontsize=15)
    boxplot.set_xticklabels(['true', 'false'])

    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'numTweets', data = graph_features_df, showfliers=True)
    boxplot.set_title("Number of Tweets (With Outliers)", fontsize=15)
    boxplot.set_xticklabels(['true', 'false'])


    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'numRetweets', data = graph_features_df, showfliers=True)
    boxplot.set_title("Number of Retweets (With Outliers)", fontsize=15)
    boxplot.set_xticklabels(['true', 'false'])


    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'retweetPercentage', data = graph_features_df, showfliers=True)
    boxplot.set_title("Retweet Percentage (With Outliers)", fontsize=15)
    boxplot.set_xticklabels(['true', 'false'])


    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'usersTouched10hours', data = graph_features_df, showfliers=True)
    boxplot.set_title("Users Touched in 10 hours (With Outliers) ", fontsize=15)
    boxplot.set_xticklabels(['true', 'false'])


    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'avgFav', data = graph_features_df, showfliers=True)
    boxplot.set_title("Average Favourite Count (With Outliers)")
    boxplot.set_xticklabels(['true', 'false'])


    # In[31]:


    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'followingAvg', data = graph_features_df, showfliers=False)
    boxplot.set_title("Average number of Followings (Without Outliers)")
    boxplot.set_xticklabels(['true', 'false'])


    # In[33]:


    sns.set_style("whitegrid")
    boxplot = sns.boxplot(x = 'label', y = 'followerAvg', data = graph_features_df, showfliers=True)
    boxplot.set_title("Average number of Followers (With Outliers)")
    boxplot.set_xticklabels(['true', 'false'])




# In[45]:




graph_features_df = sklearn.utils.shuffle(graph_features_df)
graph_features_df.head()
graph_features_df.shape

print(realGraphs.shape)
print(fakeGraphs.shape)


# ### Balance the dataset before training any classifier on it

# In[7]:



DTclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
RFclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
QUADclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
LINclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
BAYclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
LOGclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
KNNclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])
SVMclassifierScores = pd.DataFrame(columns=['ftMean', 'ftStd','stMean','stStd','accMean','accStd','precMean','precStd'
                                        ,'recMean','recStd','f1Mean','f1Std','AUCMean','AUCStd'])


# In[53]:



"""MAKE THE DATASET BALANCED between true and fake news
take the smallest size out of the 2 sets"""
realGraphs= shuffle(realGraphs)
fakeGraphs = shuffle(fakeGraphs)


finalLength=0
if(len(realGraphs) < len(fakeGraphs)):
    finalLength = len(realGraphs)
else:
    finalLength = len(fakeGraphs)
print(finalLength)


#real_graph_dataset_new = realGraphs.iloc[0:finalLength]
real_graph_dataset_new = realGraphs.sample(n=finalLength)
#fake_graph_dataset_new = fakeGraphs.iloc[0:finalLength]
fake_graph_dataset_new = fakeGraphs.sample(n=finalLength)

graph_dataset_new = pd.concat([real_graph_dataset_new, fake_graph_dataset_new])
print(graph_dataset_new.shape)
graph_dataset_new = shuffle(graph_dataset_new)
graph_dataset_new.head()


# In[208]:


print(graph_dataset_new.shape)
print(graph_features_df.shape)


# In[54]:


featureList = {'followerAvg','followingAvg','retweetPercentage','numTweets','numRetweets',
                                            'avgTimeDiff','avgFav','avgRetCount','time_first_last','usersTouched10hours','percTweets1hour'
                                  }

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
featurePowerset = list(powerset(featureList))

print(len(featurePowerset))

print(featurePowerset[100:120])

setAccuracies = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1','roc'])


# In[58]:


for i in range(1,len(featurePowerset)):
    featureSet = set(featurePowerset[i])
    print(featureSet)
    graph_as_array = graph_dataset_new[featureSet].to_numpy()
    #graph_as_array = graph_features_df[featureSet].to_numpy()
    labels_as_array = graph_dataset_new[{'label'}].to_numpy()
    #labels_as_array = graph_features_df[{'label'}].to_numpy()
    
    print('graph array size : '+str(len(graph_as_array)))
    print('label array size : '+str(len(labels_as_array)))
    
    random_forest = RandomForestClassifier()

    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(random_forest, graph_as_array, labels_as_array, scoring=scoring, cv=10)

    sorted(scores.keys())
    forest_accuracy = scores['test_accuracy'].mean()
    forest_precision = scores['test_precision_macro'].mean()
    forest_recall = scores['test_recall_macro'].mean()
    forest_f1 = scores['test_f1_weighted'].mean()
    forest_roc = scores['test_roc_auc'].mean()

    #setAccuracies = setAccuracies.append({'accuracy':forest_accuracy, 'precision':forest_precision, 'recall':forest_recall, 
     #                                    'f1':forest_f1,'roc':forest_roc}, ignore_index=True)
    print(str(featureSet))
    setAccuracies.loc[str(featureSet)] = [forest_accuracy, forest_precision, forest_recall, 
                                         forest_f1,forest_roc]


# In[57]:


from sklearn.metrics import confusion_matrix



graph_as_array = graph_features_df[{'followerAvg','followingAvg','retweetPercentage','numTweets','numRetweets',
                                            'avgTimeDiff','avgFav','avgRetCount','time_first_last','usersTouched10hours','percTweets1hour'
                     }].to_numpy()

print(len(graph_as_array))
labels_as_array = graph_features_df[{'label'}].to_numpy()
labels_as_array=labels_as_array.astype('int')
#kf = cross_validation.KFold(len(y), n_folds=10)
conf_matrix_list_of_arrays = []
random_forest = RandomForestClassifier()
for i in range(0,10):
    random_forest = RandomForestClassifier()
    training_graphs, test_graphs, training_labels, test_labels = train_test_split(graph_as_array, labels_as_array, test_size=0.3)
    random_forest.fit(training_graphs, training_labels)
    conf_matrix = confusion_matrix(test_labels, random_forest.predict(test_graphs))
    conf_matrix_list_of_arrays .append(conf_matrix)
mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
print(mean_of_conf_matrix_arrays)


# In[60]:


bestAccuraciesOverSetSizes = pd.DataFrame(columns=['bestAcc'])
bestROCOverSetSizes = pd.DataFrame(columns=['bestROC'])
bestF1OverSetSizes = pd.DataFrame(columns=['bestF1'])

sizes=range(1,12)
featuresIndicesNames = list(setAccuracies.index)

for size in sizes:
    size1scores=pd.DataFrame(columns=['accuracy','precision','recall','f1','roc'])
    print('size : '+str(size))
    for index in featuresIndicesNames:
        #print('index : '+str(index))
        setSize = index.count(',')+1
        #print("set size : "+str(setSize))
        if setSize ==size:
            #print(setAccuracies.loc[index])
            size1scores.loc[index] =[setAccuracies.loc[index].accuracy,setAccuracies.loc[index].precision, setAccuracies.loc[index].recall, setAccuracies.loc[index].f1, setAccuracies.loc[index].roc ]


    size1scores = size1scores.sort_values(by='accuracy', ascending=False)
    #size1scores = size1scores.sort_values(by='f1', ascending=False)
    #size1scores = size1scores.sort_values(by='roc', ascending=False)
    #display(size1scores)
    #size1scores.to_latex()
    #print(size1scores.iloc[0].index)
    indicesNames = list(size1scores.index)
    print(indicesNames[0])

    bestAccuraciesOverSetSizes.loc[str(size)+' : '+ indicesNames[0]] = [size1scores.iloc[0].accuracy]
    #bestF1OverSetSizes.loc[str(size)+' : '+ indicesNames[0]] = [size1scores.iloc[0].f1]
    #bestROCOverSetSizes.loc[str(size)+' : '+ indicesNames[0]] = [size1scores.iloc[0].roc]
    
#display(bestF1OverSetSizes)
display(bestAccuraciesOverSetSizes)
#display(bestROCOverSetSizes)


# In[491]:


plt.plot([1,2,3,4,5,6,7,8,9,10,11],bestF1OverSetSizes.bestF1, marker='o')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])
plt.xlabel('feature subset size', fontsize=13)
plt.ylabel('best F1 score', fontsize=13)
plt.title('Best F1 score per feature subset size', fontsize=15)
plt.show()


# In[55]:


from IPython.display import display, HTML

largest = setAccuracies.nlargest(100, 'accuracy')
display(largest)


# In[ ]:


plt.bar()


# In[56]:


#print(type(setAccuracies.nlargest(15, 'accuracy')))

largest = setAccuracies.nlargest(1024, 'accuracy')
#print(largest)
featuresIndicesNames = list(largest.index)




avgNumFeatures=0
for featureIndex in featuresIndicesNames: 
    length = featureIndex.count(',')+1
    #print('len feature index '+str(len(featureIndex)))
    #print(featureIndex)
    avgNumFeatures = avgNumFeatures+length
print(avgNumFeatures/len(featuresIndicesNames))

print(len(featuresIndicesNames))

followingCount=0
followerCount=0
tweetsCount=0
retweetsCount=0
retweetPercCount=0
timeDiffCount=0
first_lastCount=0
avgRetCountCount=0
avgFavCount=0
usersTouchedCount=0
percRetCount=0

for featureIndex in featuresIndicesNames:
    if 'followingAvg' in featureIndex:
        followingCount+=1
    if 'followerAvg' in featureIndex:
        followerCount+=1
    if 'numTweets' in featureIndex:
        tweetsCount+=1
    if 'numRetweets' in featureIndex:
        retweetsCount+=1
    if 'retweetPercentage' in featureIndex:
        retweetPercCount+=1
    if 'avgTimeDiff' in featureIndex:
        timeDiffCount+=1
    if 'time_first_last' in featureIndex:
        first_lastCount+=1
    if 'avgRetCount' in featureIndex:
        avgRetCountCount+=1
    if 'avgFav' in featureIndex:
        avgFavCount+=1
    if 'usersTouched10hours' in featureIndex: 
        usersTouchedCount+=1
    if 'percTweets1hour' in featureIndex: 
        percRetCount+=1



print('following count : '+str(followingCount))
print('follower count : '+str(followerCount))
print('tweets count : '+str(tweetsCount))
print('retweets count : '+str(retweetsCount))
print('time diff count : '+str(timeDiffCount))
print('time first last count : '+str(first_lastCount))
print('avg ret count count : '+str(avgRetCountCount))
print('avg fav count : '+str(avgFavCount))
print('users touched count : '+str(usersTouchedCount))
print('perc ret count : '+str(percRetCount))



# In[209]:


#graph_as_array = graph_dataset_new[{'numNodes','numEdges','followerAvg','followingAvg','graphDensity','retweetPercentage',
                                          #'avgTimeDiff','numTweets','numRetweets', 'avgFav', 'avgRetCount',
                                          #'time_first_last', 'edgesToNodes','nodesToEdges', 'usersTouched1day'}].to_numpy()

#graph_as_array = graph_features_df[{'avgTimeDiff'}].to_numpy()
##AFTER CORRELATION ANALYSIS + GRAPH DENSITY OUT OF THE CONTEXT
#chosenFeatures = ['followerAvg','followingAvg','retweetPercentage','avgTimeDiff','numTweets','numRetweets','avgFav', 'avgRetCount',
 #                                         'time_first_last',  'usersTouched10hours','percTweets1hour']

#graph_as_array = graph_dataset_new[{'followerAvg','followingAvg','avgTimeDiff','numTweets','numRetweets','avgFav', 'avgRetCount',
 #                                         'time_first_last', 'usersTouched10hours','percTweets1hour'}].to_numpy()

chosenFeatures = ['followerAvg','followingAvg','retweetPercentage','numTweets','numRetweets',
                                            'avgTimeDiff','avgFav','avgRetCount','time_first_last','usersTouched10hours','percTweets1hour'
                                   ]

graph_as_array = graph_dataset_new[{'followerAvg','followingAvg','retweetPercentage','numTweets','numRetweets',
                                            'avgTimeDiff','avgFav','avgRetCount','time_first_last','usersTouched10hours','percTweets1hour'
                     }].to_numpy()

#graph_as_array = graph_features_df[{'followerAvg','followingAvg','retweetPercentage','numTweets','numRetweets',
 #                                           'avgTimeDiff','avgFav','avgRetCount','time_first_last','usersTouched10hours','percTweets1hour'
  #                   }].to_numpy()

#graph_as_array = graph_dataset_new[{'time_first_last','followingAvg', 'followerAvg'}].to_numpy()



print(len(graph_as_array))
labels_as_array = graph_dataset_new[{'label'}].to_numpy()
labels_as_array=labels_as_array.astype('int')

#labels_as_array = graph_features_df[{'label'}].to_numpy()
#labels_as_array=labels_as_array.astype('int')

print(len(graph_as_array))
training_graphs, test_graphs, training_labels, test_labels = train_test_split(graph_as_array, labels_as_array, test_size=0.3)
        
print("training graphs size : "+str(len(training_graphs)))
print("training labels size : "+str(len(training_labels)))
print("test graphs size : "+str(len(test_graphs)))
print("test labels size : "+str(len(test_labels)))



# In[210]:


random_forest = RandomForestClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(random_forest, graph_as_array, labels_as_array, scoring=scoring, cv=10)

sorted(scores.keys())
forest_fit_time = scores['fit_time'].mean()
forest_score_time = scores['score_time'].mean()
forest_accuracy = scores['test_accuracy'].mean()
forest_precision = scores['test_precision_macro'].mean()
forest_recall = scores['test_recall_macro'].mean()
forest_f1 = scores['test_f1_weighted'].mean()
forest_roc = scores['test_roc_auc'].mean()

forest_fit_time_sd = scores['fit_time'].std()
forest_score_time_sd  = scores['score_time'].std()
forest_accuracy_sd  = scores['test_accuracy'].std()
forest_precision_sd  = scores['test_precision_macro'].std()
forest_recall_sd  = scores['test_recall_macro'].std()
forest_f1_sd  = scores['test_f1_weighted'].std()
forest_roc_sd  = scores['test_roc_auc'].std()

print('accuracy : '+str(forest_accuracy))


RFclassifierScores = RFclassifierScores.append({'ftMean':forest_fit_time, 'ftStd':forest_fit_time_sd,
                            'stMean':forest_score_time,'stStd':forest_score_time_sd,
                            'accMean':forest_accuracy,'accStd':forest_accuracy_sd,
                            'precMean':forest_precision,'precStd':forest_precision_sd,
                            'recMean':forest_recall,'recStd':forest_recall_sd,
                            'f1Mean':forest_f1,'f1Std':forest_f1_sd,
                            'AUCMean':forest_roc,'AUCStd':forest_roc_sd}, ignore_index=True)


# In[ ]:





# In[211]:


classifier = svm.SVR()
classifier.fit(training_graphs, training_labels)


# In[212]:


test_predictions = classifier.predict(test_graphs)
print('Accuracy of SVM classifier on test graphs: {:.2f}'.format(classifier.score(test_graphs, test_labels)))


# In[213]:


clf = RandomForestClassifier(n_jobs=2, random_state=0)
#print(type(training_graphs))
#print(training_labels)
clf.fit(training_graphs, training_labels)
test_predictions = clf.predict(test_graphs)
print('Accuracy of Random Forest classifier on test graphs: {:.2f}'.format(clf.score(test_graphs, test_labels)))


# In[196]:


importances = clf.feature_importances_

print(importances)
nameAndCoeffs = list()
for i in range(0,len(importances)):
    nameAndCoeff = [chosenFeatures[i], importances[i]]
    nameAndCoeffs.append(nameAndCoeff)

print(nameAndCoeffs)


# In[197]:


import matplotlib.pyplot as plt

std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
indicesNames=list()

for i in range(0,len(indices)):
    indicesNames.append(chosenFeatures[indices[i]])
print(indicesNames)

# Print the feature ranking
print("Feature ranking:")

for f in range(training_graphs.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(training_graphs.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(training_graphs.shape[1]), indicesNames, rotation='vertical')
plt.xlim([-1, training_graphs.shape[1]])
plt.show()


# In[62]:


plt.bar(chosenFeatures, importances)


# In[100]:


from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(clf, threshold=0.15)
sfm.fit(training_graphs, training_labels)


# In[101]:


for feature in zip(chosenFeatures, clf.feature_importances_):
    print(feature)


# ## Librairies to download

# In[56]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV



LR = LogisticRegression()
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(LR, graph_as_array, labels_as_array, scoring=scoring, cv=10)
sorted(scores.keys())
LR_fit_time = scores['fit_time'].mean()
LR_score_time = scores['score_time'].mean()
LR_accuracy = scores['test_accuracy'].mean()
LR_precision = scores['test_precision_macro'].mean()
LR_recall = scores['test_recall_macro'].mean()
LR_f1 = scores['test_f1_weighted'].mean()
LR_roc = scores['test_roc_auc'].mean()

LR_fit_time_sd = scores['fit_time'].std()
LR_score_time_sd = scores['score_time'].std()
LR_accuracy_sd = scores['test_accuracy'].std()
LR_precision_sd = scores['test_precision_macro'].std()
LR_recall_sd = scores['test_recall_macro'].std()
LR_f1_sd = scores['test_f1_weighted'].std()
LR_roc_sd = scores['test_roc_auc'].std()

print('accuracy : '+str(LR_accuracy))
print(LR_fit_time,LR_score_time,LR_accuracy,LR_precision,LR_recall,LR_f1,LR_roc)
print(LR_fit_time_sd,LR_score_time_sd,LR_accuracy_sd,LR_precision_sd,LR_recall_sd,LR_f1_sd,LR_roc_sd)

LOGclassifierScores = LOGclassifierScores.append({'ftMean':LR_fit_time, 'ftStd':LR_fit_time_sd,
                            'stMean':LR_score_time,'stStd':LR_score_time_sd,
                            'accMean':LR_accuracy,'accStd':LR_accuracy_sd,
                            'precMean':LR_precision,'precStd':LR_precision_sd,
                            'recMean':LR_recall,'recStd':LR_recall_sd,
                            'f1Mean':LR_f1,'f1Std':LR_f1_sd,
                            'AUCMean':LR_roc,'AUCStd':LR_roc_sd}, ignore_index=True)


# ## Decision Treee classifier

# In[215]:


decision_tree = DecisionTreeClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(decision_tree, graph_as_array, labels_as_array, scoring=scoring, cv=10)


sorted(scores.keys())
dtree_fit_time = scores['fit_time'].mean()
dtree_score_time = scores['score_time'].mean()
dtree_accuracy = scores['test_accuracy'].mean()
dtree_precision = scores['test_precision_macro'].mean()
dtree_recall = scores['test_recall_macro'].mean()
dtree_f1 = scores['test_f1_weighted'].mean()
dtree_roc = scores['test_roc_auc'].mean()

dtree_fit_time_sd = scores['fit_time'].std()
dtree_score_time_sd = scores['score_time'].std()
dtree_accuracy_sd = scores['test_accuracy'].std()
dtree_precision_sd = scores['test_precision_macro'].std()
dtree_recall_sd = scores['test_recall_macro'].std()
dtree_f1_sd = scores['test_f1_weighted'].std()
dtree_roc_sd = scores['test_roc_auc'].std()

print('accuracy : '+str(dtree_accuracy))

DTclassifierScores = DTclassifierScores.append({'ftMean':dtree_fit_time, 'ftStd':dtree_fit_time_sd,
                            'stMean':dtree_score_time,'stStd':dtree_score_time_sd,
                            'accMean':dtree_accuracy,'accStd':dtree_accuracy_sd,
                            'precMean':dtree_precision,'precStd':dtree_precision_sd,
                            'recMean':dtree_recall,'recStd':dtree_recall_sd,
                            'f1Mean':dtree_f1,'f1Std':dtree_f1_sd,
                            'AUCMean':dtree_roc,'AUCStd':dtree_roc_sd}, ignore_index=True)


# ## SVM classifier

# In[216]:


SVM = SVC(probability = True)
#SVM = LinearSVC()

scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(SVM, graph_as_array, labels_as_array, scoring=scoring, cv=10)
    
sorted(scores.keys())
SVM_fit_time = scores['fit_time'].mean()
SVM_score_time = scores['score_time'].mean()
SVM_accuracy = scores['test_accuracy'].mean()
SVM_precision = scores['test_precision_macro'].mean()
SVM_recall = scores['test_recall_macro'].mean()
SVM_f1 = scores['test_f1_weighted'].mean()
SVM_roc = scores['test_roc_auc'].mean()

SVM_fit_time_sd = scores['fit_time'].std()
SVM_score_time_sd = scores['score_time'].std()
SVM_accuracy_sd = scores['test_accuracy'].std()
SVM_precision_sd = scores['test_precision_macro'].std()
SVM_recall_sd = scores['test_recall_macro'].std()
SVM_f1_sd = scores['test_f1_weighted'].std()
SVM_roc_sd = scores['test_roc_auc'].std()

print('accuracy : '+str(SVM_accuracy))


SVMclassifierScores = SVMclassifierScores.append({'ftMean':SVM_fit_time, 'ftStd':SVM_fit_time_sd,
                            'stMean':SVM_score_time,'stStd':SVM_score_time_sd,
                            'accMean':SVM_accuracy,'accStd':SVM_accuracy_sd,
                            'precMean':SVM_precision,'precStd':SVM_precision_sd,
                            'recMean':SVM_recall,'recStd':SVM_recall_sd,
                            'f1Mean':SVM_f1,'f1Std':SVM_f1_sd,
                            'AUCMean':SVM_roc,'AUCStd':SVM_roc_sd}, ignore_index=True)


SVC.get_params(SVC)


# ## Linear Discriminant Analysis

# In[217]:


LDA = LinearDiscriminantAnalysis()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(LDA, graph_as_array, labels_as_array,scoring=scoring, cv=9)

sorted(scores.keys())
LDA_fit_time = scores['fit_time'].mean()
LDA_score_time = scores['score_time'].mean()
LDA_accuracy = scores['test_accuracy'].mean()
LDA_precision = scores['test_precision_macro'].mean()
LDA_recall = scores['test_recall_macro'].mean()
LDA_f1 = scores['test_f1_weighted'].mean()
LDA_roc = scores['test_roc_auc'].mean()

LDA_fit_time_sd = scores['fit_time'].std()
LDA_score_time_sd = scores['score_time'].std()
LDA_accuracy_sd = scores['test_accuracy'].std()
LDA_precision_sd = scores['test_precision_macro'].std()
LDA_recall_sd = scores['test_recall_macro'].std()
LDA_f1_sd = scores['test_f1_weighted'].std()
LDA_roc_sd = scores['test_roc_auc'].std()

print('accuracy : '+str(LDA_accuracy))



LINclassifierScores = LINclassifierScores.append({'ftMean':LDA_fit_time, 'ftStd':LDA_fit_time_sd,
                            'stMean':LDA_score_time,'stStd':LDA_score_time_sd,
                            'accMean':LDA_accuracy,'accStd':LDA_accuracy_sd,
                            'precMean':LDA_precision,'precStd':LDA_precision_sd,
                            'recMean':LDA_recall,'recStd':LDA_recall_sd,
                            'f1Mean':LDA_f1,'f1Std':LDA_f1_sd,
                            'AUCMean':LDA_roc,'AUCStd':LDA_roc_sd}, ignore_index=True)


# ## Quadratic Discriminant Analysis

# In[218]:


QDA = QuadraticDiscriminantAnalysis()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(QDA, graph_as_array, labels_as_array,scoring=scoring, cv=10)

sorted(scores.keys())
QDA_fit_time = scores['fit_time'].mean()
QDA_score_time = scores['score_time'].mean()
QDA_accuracy = scores['test_accuracy'].mean()
QDA_precision = scores['test_precision_macro'].mean()
QDA_recall = scores['test_recall_macro'].mean()
QDA_f1 = scores['test_f1_weighted'].mean()
QDA_roc = scores['test_roc_auc'].mean()

QDA_fit_time_sd = scores['fit_time'].std()
QDA_score_time_sd = scores['score_time'].std()
QDA_accuracy_sd = scores['test_accuracy'].std()
QDA_precision_sd = scores['test_precision_macro'].std()
QDA_recall_sd = scores['test_recall_macro'].std()
QDA_f1_sd = scores['test_f1_weighted'].std()
QDA_roc_sd = scores['test_roc_auc'].std()

print('accuracy : '+str(QDA_accuracy))

QUADclassifierScores = QUADclassifierScores.append({'ftMean':QDA_fit_time, 'ftStd':QDA_fit_time_sd,
                            'stMean':QDA_score_time,'stStd':QDA_score_time_sd,
                            'accMean':QDA_accuracy,'accStd':QDA_accuracy_sd,
                            'precMean':QDA_precision,'precStd':QDA_precision_sd,
                            'recMean':QDA_recall,'recStd':QDA_recall_sd,
                            'f1Mean':QDA_f1,'f1Std':QDA_f1_sd,
                            'AUCMean':QDA_roc,'AUCStd':QDA_roc_sd}, ignore_index=True)


# ## Random Forest Classifier

# In[219]:


random_forest = RandomForestClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(random_forest, graph_as_array, labels_as_array, scoring=scoring, cv=10)

sorted(scores.keys())
forest_fit_time = scores['fit_time'].mean()
forest_score_time = scores['score_time'].mean()
forest_accuracy = scores['test_accuracy'].mean()
forest_precision = scores['test_precision_macro'].mean()
forest_recall = scores['test_recall_macro'].mean()
forest_f1 = scores['test_f1_weighted'].mean()
forest_roc = scores['test_roc_auc'].mean()

forest_fit_time_sd = scores['fit_time'].std()
forest_score_time_sd  = scores['score_time'].std()
forest_accuracy_sd  = scores['test_accuracy'].std()
forest_precision_sd  = scores['test_precision_macro'].std()
forest_recall_sd  = scores['test_recall_macro'].std()
forest_f1_sd  = scores['test_f1_weighted'].std()
forest_roc_sd  = scores['test_roc_auc'].std()

print('accuracy : '+str(forest_accuracy))


RFclassifierScores = RFclassifierScores.append({'ftMean':forest_fit_time, 'ftStd':forest_fit_time_sd,
                            'stMean':forest_score_time,'stStd':forest_score_time_sd,
                            'accMean':forest_accuracy,'accStd':forest_accuracy_sd,
                            'precMean':forest_precision,'precStd':forest_precision_sd,
                            'recMean':forest_recall,'recStd':forest_recall_sd,
                            'f1Mean':forest_f1,'f1Std':forest_f1_sd,
                            'AUCMean':forest_roc,'AUCStd':forest_roc_sd}, ignore_index=True)


# ## K neighbors classifier

# In[220]:


KNN = KNeighborsClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(KNN, graph_as_array, labels_as_array, scoring=scoring, cv=10)

sorted(scores.keys())
KNN_fit_time = scores['fit_time'].mean()
KNN_score_time = scores['score_time'].mean()
KNN_accuracy = scores['test_accuracy'].mean()
KNN_precision = scores['test_precision_macro'].mean()
KNN_recall = scores['test_recall_macro'].mean()
KNN_f1 = scores['test_f1_weighted'].mean()
KNN_roc = scores['test_roc_auc'].mean()

KNN_fit_time_sd = scores['fit_time'].std()
KNN_score_time_sd = scores['score_time'].std()
KNN_accuracy_sd = scores['test_accuracy'].std()
KNN_precision_sd = scores['test_precision_macro'].std()
KNN_recall_sd = scores['test_recall_macro'].std()
KNN_f1_sd = scores['test_f1_weighted'].std()
KNN_roc_sd = scores['test_roc_auc'].std()


KNNclassifierScores = KNNclassifierScores.append({'ftMean':KNN_fit_time, 'ftStd':KNN_fit_time_sd,
                            'stMean':KNN_score_time,'stStd':KNN_score_time_sd,
                            'accMean':KNN_accuracy,'accStd':KNN_accuracy_sd,
                            'precMean':KNN_precision,'precStd':KNN_precision_sd,
                            'recMean':KNN_recall,'recStd':KNN_recall_sd,
                            'f1Mean':KNN_f1,'f1Std':KNN_f1_sd,
                            'AUCMean':KNN_roc,'AUCStd':KNN_roc_sd}, ignore_index=True)
print('accuracy : '+str(KNN_accuracy))


# ## Naive Bayes

# In[221]:


bayes = GaussianNB()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(bayes, graph_as_array, labels_as_array, scoring=scoring, cv=10)

sorted(scores.keys())
bayes_fit_time = scores['fit_time'].mean()
bayes_score_time = scores['score_time'].mean()
bayes_accuracy = scores['test_accuracy'].mean()
bayes_precision = scores['test_precision_macro'].mean()
bayes_recall = scores['test_recall_macro'].mean()
bayes_f1 = scores['test_f1_weighted'].mean()
bayes_roc = scores['test_roc_auc'].mean()

bayes_fit_time_sd = scores['fit_time'].std()
bayes_score_time_sd = scores['score_time'].std()
bayes_accuracy_sd = scores['test_accuracy'].std()
bayes_precision_sd = scores['test_precision_macro'].std()
bayes_recall_sd = scores['test_recall_macro'].std()
bayes_f1_sd = scores['test_f1_weighted'].std()
bayes_roc_sd = scores['test_roc_auc'].std()


BAYclassifierScores = BAYclassifierScores.append({'ftMean':bayes_fit_time, 'ftStd':bayes_fit_time_sd,
                            'stMean':bayes_score_time,'stStd':bayes_score_time_sd,
                            'accMean':bayes_accuracy,'accStd':bayes_accuracy_sd,
                            'precMean':bayes_precision,'precStd':bayes_precision_sd,
                            'recMean':bayes_recall,'recStd':bayes_recall_sd,
                            'f1Mean':bayes_f1,'f1Std':bayes_f1_sd,
                            'AUCMean':bayes_roc,'AUCStd':bayes_roc_sd}, ignore_index=True)

print('accuracy : '+str(bayes_accuracy))


# In[222]:


models_correlation = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],
    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],
    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],
    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],
    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],
    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],
    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],
    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],
    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])

models_correlation.sort_values(by='Accuracy', ascending=False)


# # Average the scores over the 10 different random 'data set balancing'

# In[71]:


print(DTclassifierScores.shape)
print(RFclassifierScores.shape)
print(QUADclassifierScores.shape)
print(LINclassifierScores.shape)
print(BAYclassifierScores.shape)
print(LOGclassifierScores.shape)
print(KNNclassifierScores.shape)
print(SVMclassifierScores.shape)


# In[72]:


a = RFclassifierScores.accMean.mean()
print(RFclassifierScores.accMean.std())
print(RFclassifierScores.accStd.mean())
print(RFclassifierScores.accStd.std())
models_correlation = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],
    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],
    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],
    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],
    'Precision'   : [LR_precision, dtree_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],
    'Recall'      : [LR_recall, dtree_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],
    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],
    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],
    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])


"""type2 = pd.DataFrame({
    'Score'     : ['Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'],
    'AvgMean'   : [1,1,1,1,1],
    'AvgMeanStd': [2,2,2,2,2],
    'AvgStd'    : [3,3,3,3,3],
    'AvgStdStd' : [4,4,4,4,4],
    }, columns = ['Score', 'AvgMean', 'AvgMeanStd', 'AvgStd', 'AvgStdStd'])
"""
                  
RF_overall_perf = pd.DataFrame({
    'Score': ['Accuracy', 'Precision','Recall','F1_score','AUC_ROC'],
    'AvgMean': [RFclassifierScores.accMean.mean() , RFclassifierScores.precMean.mean() ,RFclassifierScores.recMean.mean(), RFclassifierScores.f1Mean.mean(), RFclassifierScores.AUCMean.mean()],
    'AvgMeanStd': [RFclassifierScores.accMean.std() ,RFclassifierScores.precMean.std() , RFclassifierScores.recMean.std(), RFclassifierScores.f1Mean.std(), RFclassifierScores.AUCMean.std()],
    'AvgStd' : [RFclassifierScores.accStd.mean(), RFclassifierScores.precStd.mean(), RFclassifierScores.recStd.mean(), RFclassifierScores.f1Std.mean(), RFclassifierScores.AUCStd.mean()],
    'AvgStdStd':[RFclassifierScores.accStd.std() ,RFclassifierScores.precStd.std() ,RFclassifierScores.recStd.std(),RFclassifierScores.f1Std.std(),RFclassifierScores.AUCStd.std()],
}, columns = ['Score', 'AvgMean', 'AvgMeanStd', 'AvgStd', 'AvgStdStd'])

RF_overall_perf.head()

#type2.head()


# ## Voting classifier

# In[48]:


models = [LogisticRegression(),
         DecisionTreeClassifier(),
         SVC(probability = True),
         LinearDiscriminantAnalysis(),
         QuadraticDiscriminantAnalysis(),
         RandomForestClassifier(),
         KNeighborsClassifier(),
         GaussianNB()]

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']


# In[49]:


for model in models:
    scores = cross_validate(model, training_graphs, training_labels, scoring=scoring, cv=9)


# ### HARD

# In[134]:


models_ens = list(zip(['LR', 'DT', 'SVM', 'LDA', 'QDA', 'RF', 'KNN', 'NB'], models))


model_ens = VotingClassifier(estimators = models_ens, voting = 'hard')
model_ens.fit(training_graphs, training_labels)
pred = model_ens.predict(test_graphs)

acc_hard = accuracy_score(test_labels, pred)
prec_hard = precision_score(test_labels, pred)
recall_hard = recall_score(test_labels, pred)
f1_hard = f1_score(test_labels, pred)
roc_auc_hard = 'not applicable'


# ### SOFT

# In[135]:


model_ens = VotingClassifier(estimators = models_ens, voting = 'soft')
model_ens.fit(training_graphs, training_labels)
pred = model_ens.predict(test_graphs)
prob = model_ens.predict_proba(test_graphs)[:,1]

acc_soft = accuracy_score(test_labels, pred)
prec_soft = precision_score(test_labels, pred)
recall_soft = recall_score(test_labels, pred)
f1_soft = f1_score(test_labels, pred)
roc_auc_soft = roc_auc_score(test_labels, prob)


# In[136]:


models_ensembling = pd.DataFrame({
    'Model'       : ['Ensebling_hard', 'Ensembling_soft'],
    'Accuracy'    : [acc_hard, acc_soft],
    'Precision'   : [prec_hard, prec_soft],
    'Recall'      : [recall_hard, recall_soft],
    'F1_score'    : [f1_hard, f1_soft],
    'AUC_ROC'     : [roc_auc_hard, roc_auc_soft],
    }, columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])

models_ensembling.sort_values(by='Accuracy', ascending=False)


# ### VISUAL CREATION

# In[17]:


graph_dataset=list()
graph_features_df = pd.DataFrame(columns=['numNodes','numEdges','followerAvg','followingAvg','graphDensity','retweetPercentage',
                                          'avgTimeDiff','numTweets','numRetweets', 'avgFav', 'avgRetCount',
                                          'time_first_last', 'edgesToNodes','nodesToEdges', 'usersTouched1day',
                                          'percRet1hour','label'])


from scipy.stats import ttest_ind
pvalues = list()
tStats=list()
for feature in graph_features_df:
    if feature != 'label':
        stats = ttest_ind(realGraphs[feature], fakeGraphs[feature])
        pvalue = "{0:.7f}".format(stats[1])
        tstat = "{0:.7f}".format(stats[0])
        print('p value : '+pvalue)
        print('t : '+tstat)
        tStats.append(tstat)
        pvalues.append(str(pvalue))
print(len(tStats))
print(len(pvalues))



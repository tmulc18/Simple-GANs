import os
# import requests
from six.moves.urllib.request import urlretrieve
import pandas as pd
import numpy as np

loc = 'Data/'
filename='mnist_train.csv'
def downloadData(filename,loc):
    data_point = 'https://pjreddie.com/media/files/mnist_train.csv'
    if not os.path.exists(loc+filename):
        os.chdir(loc)
        urlretrieve(data_point, filename)
#         requests.get(data_point)
        os.chdir('..')

def getData(filename,loc):
    train_set = pd.read_csv(loc+filename,header=None)
    #get labels in own array
    train_lb=np.array(train_set[0])

    #one hot encode the labels
    train_lb=(np.arange(10) == train_lb[:,None]).astype(np.float32)

    #drop the labels column from training dataframe
    trainX=train_set.drop(0,axis=1)

    #put in correct float32 array format
    trainX=np.array(trainX).astype(np.float32)

    trainX=trainX.reshape(len(trainX),28,28,1)
    
    X,y = trainX, train_lb
    
    return X,y
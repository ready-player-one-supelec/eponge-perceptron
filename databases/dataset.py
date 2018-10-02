import idx2numpy
import numpy as np 


def conversion3Dto2D(ndarr):
    k,n,m = ndarr.shape
    res = np.zeros((n*m, k))
    for l in range(k):
        res[:,l] = ndarr[l,:,:].flatten()

    return res

def conversionLabels(arr):
    n = len(arr)
    res = np.zeros((10,n))
    for i in range(n):
        nb = arr[i]
        res[nb,i] = 1
    return res

class DataSet():
    def __init__(self):
        return
    
    def returnTrainImages(self):
        return conversion3Dto2D(idx2numpy.convert_from_file('databases/train-images-idx3-ubyte'))
    
    def returnTrainLabels(self):
        return conversionLabels(idx2numpy.convert_from_file('databases/train-labels-idx1-ubyte'))

    def returnTestImages(self):
        return conversion3Dto2D(idx2numpy.convert_from_file('databases/t10k-images-idx3-ubyte'))
    
    def returnTestLabels(self):
        return conversionLabels(idx2numpy.convert_from_file('databases/t10k-labels-idx1-ubyte'))
    

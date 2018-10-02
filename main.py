import numpy as np
from classes import perceptron
from databases import dataset


print('----------------------------  Initialisation du perceptron -------------------------------------------')
ds = dataset.DataSet()

sizeTrain = 1000
sizeTest = 100
iterations = 1000


trainData = ds.returnTrainImages()[:,0:sizeTrain]
trainLabels = ds.returnTrainLabels()[:,0:sizeTrain]



testData = ds.returnTestImages()[:,0:sizeTest]
testLabels = ds.returnTestLabels()[:, 0:sizeTest]
debut = trainData.shape[0]
fin = trainLabels.shape[0]
listDimension = [debut, 1000, fin]

def activation(X):
    
    res =  np.divide(1, 1+np.exp(-X))
    return res

def activationDerivative(Z,A):
    return np.multiply(A, 1 - A)

p = perceptron.Perceptron(listDimension, trainData, testData, activation, activationDerivative, trainLabels)

print("------------------------------  Done  --------------------------------------")

print("------------------------------ Apprentissage pendant %d itérations ------------------" % iterations)

for i in range(iterations):
    if (i%100 == 0):
        print('progression : ', i*100/iterations, ' %')
        p.iterateOnce(True)
    else:
        p.iterateOnce(False)
print("------------------------------  Done  --------------------------------------")

print("------------------------------  Tests sur les données  -----------------------------")
def nbOfError(predict, result):
    res = 0
    for i in range(predict.shape[1]):
        if (np.argmax(predict[:,i], axis=0) != np.argmax(testLabels[:,i], axis=0)):
            res+=1
    return res


print("% d'erreur : " , nbOfError(p.getPrediction(), testLabels))
predict = p.getPrediction()




for i in range(predict.shape[1]):
    if ((i%10) == 0):
        print("prediction : ")
        print(np.argmax(predict[:,i], axis=0))
        print("result : ")
        print(np.argmax(testLabels[:,i], axis=0))
        print("____________")
        input()

print("------------------------------  Done  --------------------------------------")
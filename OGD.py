import numpy as np



    


def OGD_learn(trials,labels,Y):

    d = len(trials)

    #hypothesis weights
    W=[np.zeros(d) for i in range(Y)] 

    for i in range(len(trails)):
        #get prediction:

        p = max(range(10),key = lambda y:np.dot(W[y],trials[i]))

        

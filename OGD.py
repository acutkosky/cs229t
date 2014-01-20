from sys import argv
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt



def test_err(W,testset,testlabels,Y):

    hinge = 0
    zero_one = 0

    

    for t in range(len(testset)):
        label = testlabels[t]-1
        WyX = np.dot(W[label],testset[t])
        hinge += max([(y!=label) + np.dot(W[y],testset[t])-WyX for y in range(Y)])
        zero_one += (1 if (max(range(Y),key = lambda y:np.dot(W[y],testset[t]))) != label else 0)

    return (hinge/len(testset),float(zero_one)/len(testset))
        
    


def OGD_learn(trials,labels,Y,eta,best,testset,testlabels,T=None):

    if(T == None):
        T = len(trials)

    d = len(trials[0])

    zero_one_averages = np.zeros(T)
    ft_averages = np.zeros(T)
    zero_one_best_averages = np.zeros(T)
    ft_best_averages = np.zeros(T)

    #hypothesis weights
    W=[np.zeros(d) for i in range(Y)] 


    zero_one_total = 0
    ft_total = 0

    zero_one_best_total = 0
    ft_best_total = 0

    print "len labels:",len(labels)
    print "len trials:",len(trials)

    timepoints = []
    lth = []
    ltz = []
    eth = []
    etz = []
    print "Running OGD for eta = ",str(eta)
    for t in range(T):
        #get prediction
 #       if(t%1000==0):
 #           print "on iteration: "+str(t)+" for eta = "+str(eta)
        p = max(range(Y),key = lambda y:np.dot(W[y],trials[t]))

        label = labels[t]-1

        WyX = np.dot(W[label],trials[t])

        WyX_best = np.dot(best[label],trials[t])
        y_for_ft = 0
        ft = (0!=label) + np.dot(W[0],trials[t])-WyX
        ft_best=(0!=label)+np.dot(best[0],trials[t])-WyX_best
 #       print "norm: ",WyX#min([np.linalg.norm(W[y]) for y in range(Y)])

        for y in range(Y):
            temp = (y!=label) + np.dot(W[y],trials[t])-WyX
            temp_best = (y!=label)+np.dot(best[y],trials[t])-WyX_best

            if(temp>ft):
                ft = temp
                y_for_ft = y

            if(temp_best>ft_best):
                ft_best = temp_best

#        print "ft: ",ft
        if(ft == 0):
            assert(y_for_ft == label)
#                print "ft: ",ft
#                print "ft recalc: ",(y_for_ft!=label) +np.dot(W[y_for_ft],trials[t])-WyX
#                print "fail fail: yf: ",y_for_ft
#                print "fail fail  yt: ",label

        zero_one = (1 if (max(range(Y),key = lambda y:np.dot(W[y],trials[t]))) != label else 0)
        zero_one_best = (1 if (max(range(Y),key = lambda y:np.dot(best[y],trials[t]))) != label else 0)
            
        W[y_for_ft] -= eta*trials[t]
        W[label] += eta*trials[t]



        
        ft_total += ft
        zero_one_total += zero_one

        ft_best_total += ft_best
        zero_one_best_total += zero_one_best

        ft_averages[t] = ft_total/float(t+1)
        zero_one_averages[t] = zero_one_total/float(t+1)

        ft_best_averages[t] =ft_best_total/float(t+1)
        zero_one_best_averages[t] = zero_one_best_total/float(t+1)


        if(t%1000 == 0):
#            print "Calculating test error"
            lte = test_err(W,testset,testlabels,Y)
            ete = test_err(best,testset,testlabels,Y)

            lth.append(lte[0])
            ltz.append(lte[1])

            eth.append(ete[0])
            etz.append(ete[1])

            timepoints.append(t)
#            print "done!"



    print "Expert training Hinge loss: ",ft_best_averages[t]
    print "Expert training Zero-One Loss: ",zero_one_best_averages[t]
    print "Learner training Hinge Loss: ",ft_averages[t]
    print "Learner training Zero-One Loss: ",zero_one_averages[t]
    tvals = np.arange(1,T+1)
    plt.clf()



    plt.plot(tvals,ft_averages,label="Hinge Loss, Learner")
    
    plt.plot(tvals,ft_best_averages,label="Hinge Loss, Expert")
    plt.plot(tvals,zero_one_averages,label = "Zero-One Loss, Learner")
    plt.plot(tvals,zero_one_best_averages,label ="Zero-One Loss, Expert")

    plt.title("Average Hinge and Zero-One Losses, eta = "+str(eta))
    plt.legend()
    plt.savefig(("figs/lossplots_eta"+str(eta)).replace(".","p"))
    plt.ylim(0,0.5)
    plt.savefig(("figs/lossplots_zoomedin_eta"+str(eta)).replace(".","p"))

 #   plt.show()


    print "Expert test Hinge Loss: ",eth[0]
    print "Expert test Zero-One Loss: ",etz[0]
    print "Learner test Hinge Loss: ",lth[len(lth)-1]
    print "Learner test Zero-One Loss: ",ltz[len(ltz)-1]
    plt.clf()


    plt.plot(timepoints,lth,label ="Test Hinge Loss, Learner")
    plt.plot(timepoints,ltz,label = "Test Zero-One Loss, Learner")
    plt.plot(timepoints,eth,label="Test Hinge Loss, Expert")
    plt.plot(timepoints,etz,label = "Test Zero-One Loss, Expert")
    plt.title("Test Loss, eta = "+str(eta))
    plt.legend()

    plt.savefig(("figs/testloss_eta"+str(eta)).replace(".","p"))
    plt.ylim(0,0.5)
    plt.savefig(("figs/testloss_zoomedin_eta"+str(eta)).replace(".","p"))

#    plt.show()


def setup(matrixfile,Y,eta,bestfile,testfile):
    M = loadmat(matrixfile)
    best = loadmat(bestfile)['U']
    trials = M['X'].transpose()
    labels = M['y'][0]

    testM = loadmat(testfile)
    testset = testM['Xtest'].transpose()

    testlabels = testM['ytest'][0]

    print "starting to learn!"
    OGD_learn(trials,labels,Y,eta,best,testset,testlabels)



def main(args):
    setup(args[1],10,float(args[3]),args[2],args[4])


if __name__=="__main__":
    if(len(argv)<5):
        print "OGD.py matrixfile bestfile eta testfile"
    main(argv)

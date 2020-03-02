#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
import math
import time

# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for k in range(Nclasses):
        prior[k] = len(np.where(labels == k))/np.sum(W[np.where(labels == k)])
    prior = prior/np.sum(prior)
    # ==========================

    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for k in range(Nclasses):
        w_k = np.sum(W[np.where(labels==k)])
        #print(W[np.where(labels==k)].shape)
        #print(X[np.where(labels==k)].shape)
        mu[k, :] = np.sum(W[np.where(labels==k)] * X[np.where(labels==k)], axis=0)/w_k 
        for m in range(Ndims):
            sigma[k, m, m] =  np.sum(W[np.where(labels==k)].reshape(1,-1) * np.power((X[np.where(labels==k), m] - mu[k,m]), 2), axis=1)/w_k
    # ==========================

    return mu, sigma

# def mlParams(X, labels, W=None):
#     assert(X.shape[0]==labels.shape[0])
#     Npts,Ndims = np.shape(X)
#     classes = np.unique(labels)
#     Nclasses = np.size(classes)

#     if W is None:
#         W = np.ones((Npts,1))/float(Npts)

#     mu = np.zeros((Nclasses,Ndims))
#     sigma = np.zeros((Nclasses,Ndims,Ndims))

#     # TODO: fill in the code to compute mu and sigma!
#     # ==========================
#     for k in range(Nclasses):
#         N_k = len(X[np.where(labels==k)])
#         mu[k, :] = np.sum(X[np.where(labels==k)], axis=0)/N_k 
#         for m in range(Ndims):
#             sigma[k, m, m] =  np.sum(np.power( (X[np.where(labels==k), m] - mu[k,m]), 2), axis=1)/N_k
#     # ==========================

#     return mu, sigma







# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for k in range(Nclasses):
        
        det_sigma = np.linalg.det(sigma[k])
        sigma_diag = np.diag(np.linalg.inv(sigma[k]))
        X_mu = X - mu[k]

        term_1 = -0.5*math.log(det_sigma)
        term_2 = -0.5*np.sum(np.power(X_mu,2)*sigma_diag, axis=1)
        term_3 = math.log(prior[k])

        logProb[k, :] = term_1 + term_2 + term_3
    # ==========================
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.


# X, labels = genBlobs(centers=5)
# mu, sigma = mlParams(X,labels)
# plotGaussian(X,labels,mu,sigma)


# # Call the `testClassifier` and `plotBoundary` functions for this part.


# testClassifier(BayesClassifier(), dataset='iris', split=0.7)



# testClassifier(BayesClassifier(), dataset='vowel', split=0.7)



# plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)
    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        wCur = wCur.reshape(1,-1).reshape(-1)
        error = np.dot(wCur, 1 - np.where(vote == labels, 1, 0))
        alpha = 0.5*(np.log(1-error) - np.log(error))
        wCur = wCur * np.where(vote == labels, np.exp(-alpha), np.exp(alpha))
        wCur = wCur/np.sum(wCur)
        wCur = wCur.reshape(-1,1)
        alphas.append(alpha) # you will need to append the new alpha
    
        # ==========================
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)
    
    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))


        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        t = 0
        for clasifier in classifiers:
            for k in range(Nclasses):
                votes[:,k] += (alphas[t]*np.where(clasifier.classify(X) == k,1,0))
            t += 1
        
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)



# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



# plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


# testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)





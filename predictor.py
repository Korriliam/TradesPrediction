# from __future__ import unicode_literals
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from StringIO import StringIO
import traceback


__author__ = 'korrigan'

import pandas as pd
import cPickle
import numpy as np
from matplotlib.pyplot import figure, show, plot, savefig, legend
import dataRecovery
from datetime import datetime, date, time, timedelta
from math import ceil
from pybrain.datasets import SupervisedDataSet
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import cPickle as pickle
from datetime import datetime
import random
import os
import time
import itertools


from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.optimize import curve_fit
from scipy import array
from collections import defaultdict

import Orange
from Orange import tuning
from Orange.classification import svm
from Orange.classification import neural
from Orange import ensemble
# from nltk.stem import SnowballStemmer
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
from Orange.evaluation import testing, scoring
import operator


class predictor(object):

    debug = False  # niveau 3
    debugBis = True  # niveau 2
    debugTer = True  # niveau 1



    availableClassifiersPerInstance = {
        "SVM": False,
        "SVM2": False,
        "NaiveBayes": False,
        "NeuralNetwork": False,
        "Knn": False,
        "LinearSVM": True,
        "CN2": False,
        "CN2Unordered": False, 
        "BTree": False,
        "STree": False,
        "Bag": False,
        "RForest": False,
        "Stacking": False
    }

    """
    Nous allons "eduquer" notre classifieur avec les instances contenues dans la base d'apprentissage
    """
    BaseApprentissage = {}
    BaseValidation = {}

    """
    Contient la liste des instances des classifieurs
    """
    classifiers = {}

    """
    Accueille les donnes d'apprentissage.
    """
    TrainData = None

    """
    Stocke les donnes de validation
    """
    ValidationData = None


    ratioUsed = 0.6


    """
    neural network
    """
    nbNode = 25
    regFact = 0.01
    nbnnIt = 1000

    def __init__(self, symbol_of_stock, range_of_prediction=8, training_range = 500): 

        fromDateTrain = datetime.utcnow() - timedelta(days=(training_range+range_of_prediction))#on donne le nb de jour avant auj ou on commence
        toDateTrain = datetime.utcnow() - timedelta(days=range_of_prediction)#on s'arrete tant de j avan auj

        fromDateValidation = datetime.utcnow() - timedelta(days=range_of_prediction)#on donne le nb de jour avant auj ou on commence
        toDateValidation = datetime.utcnow()#on s'arrete tant de j avan auj

        self.numberOfDimensionOfDescriptor = 8

        # self.dataRecovery = dataRecovery.dataRecovery()
        # self.trainData = self.dataRecovery.yahooDownloader(symbol_of_stock, fromDateTrain, toDateTrain)
        # self.ValidationData = self.dataRecovery.yahooDownloader(symbol_of_stock, fromDateValidation, toDateValidation)
        # self.postTreatmentDataset()
        self.nbj = 60
        ssss = dataRecovery.historical_quotes(symbol_of_stock,"20050505","20140606")
        self.Data = dataRecovery.from_google_historical(symbol_of_stock,"2005-05-05")
        print u"Date size : %s" % self.Data.size
        if len(self.Data) == 0:
            print "data est vide"
            exit()
        self.defineDomain()
        self.postTreatmentDataset(self.Data)
        self.dataOrange = Orange.data.Table(self.domain, self.dataNumpy)
        # print "ok"
        learners = [
            neural.NeuralNetworkLearner(n_mid=250, reg_fact=0.01, max_iter=10000),
            Orange.classification.bayes.NaiveLearner(),
            Orange.classification.majority.MajorityLearner(),
            Orange.classification.tree.SimpleTreeLearner(min_instances=20),
            Orange.ensemble.forest.RandomForestLearner(),
            Orange.classification.svm.LinearSVMLearner(solver_type=Orange.classification.svm.LinearSVMLearner.L2R_L2LOSS_DUAL,
                                                               C=1.0,
                                                               eps=1,
                                                               normalization=True),
        ]
        cv = Orange.evaluation.testing.cross_validation(learners, self.dataOrange, folds=5)

        accuTab = Orange.evaluation.scoring.CA(cv)
        aucTab = Orange.evaluation.scoring.AUC(cv)

        print "Accuracy:",
        print ["%.4f" % score for score in accuTab]
        print "AUC:",
        print ["%.4f" % score for score in aucTab]
        print "Ended."

    def postTreatmentDataset(self, quotes):
        '''
        Va generer les donnes "pen,High,Low,Close,Volume" 5 jours, plus le label (si le 6e jour ca monte ou pas)
        :return:
        '''
        # Date Open High Low Close Volume
        variations = []
        size = quotes.shape[0]
        l = range(size-100)
        quotes = quotes[::-1] # on inverse
        # oo = np.zeros(ceil(size/10))
        u = 0
        for i in l[::self.nbj]:
            j = 0
            # print "---"
            # vec = np.zeros((self.nbj-1)*5+1)
            vec = np.zeros((self.nbj-1)+1)
            rr = self.gainNormaliseParJour(quotes[i:i+self.nbj-1])
            while j != self.nbj-1:
                # print quotes[i+j]['\xef\xbb\xbfDate']
                vec[j] = rr[j]
                # vec[j*5 + 1] = quotes[i+j]['High']
                # vec[j*5 + 2] = quotes[i+j]['Low']
                # vec[j*5 + 3] = quotes[i+j]['Close']
                # vec[j*5 + 4] = quotes[i+j]['Volume']
                j += 1
            # print "to predict : " + str(quotes[i+j]['\xef\xbb\xbfDate']) + ",",
            # on determine si a mont ou baiss

            if not isinstance(quotes[i+j]['Close'], np.float):
                a = np.float64(quotes[i+j]['Close'])
            else:
                a = quotes[i+j]['Close']
            if not isinstance(quotes[i+j]['Open'], np.float):
                b = np.float64(quotes[i+j]['Open'])
            else:
                b = quotes[i+j]['Open']

            if a - b > 0:
                # print "Up"
                vec[(self.nbj-1)] = 1
                # oo[u] = "Up"
            else:
                # print "Down"
                vec[(self.nbj-1)] = 0
                # oo[u] = "Down"
            if i == 0:
                accu = vec
            else:
                accu = np.vstack((accu,vec))
            u += 1
            # accu = np.hstack(accu,oo)
        self.dataNumpy = accu



    def defineDomain(self):
        classattr = Orange.feature.Discrete("class", values=['Up','Down'])
        features = []
        i = 0
        while i != self.nbj-1:
            features.append(Orange.feature.Continuous("Open%d" % i))
            # features.append(Orange.feature.Continuous("High%d" % i))
            # features.append(Orange.feature.Continuous("Low%d" % i))
            # features.append(Orange.feature.Continuous("Close%d" % i))
            # features.append(Orange.feature.Continuous("Volume%d" % i))
            i += 1
        self.domain = Orange.data.Domain(features + [classattr])


    def gainNormaliseParJour(self, data):
        var = []
        for elmt in data:
            try:
                var.append(elmt["Close"] - elmt["Open"])
            except:
                try:
                    ee = np.float(elmt["Close"])
                    rr = np.float(elmt["Open"])
                    var.append(ee-rr)
                except:
                    pass
        # on normalize
        a=max(var)
        o = []
        for elt in var:
            o.append(elt/a)
        return o


    def movingaverage(self, interval, window_size):
        '''
        Compute the average in a range of 5 arround the price.
        :param interval:
        :param window_size:
        :return:
        '''
        window = np.ones(int(window_size))/float(window_size)
        avg = np.convolve(interval, window, 'same')

        half_window = ceil(window_size/2)

        avg[:half_window] = interval[:half_window]
        avg[-half_window:] = interval[-half_window:]
        return avg


if __name__ == '__main__':


    tab = pd.read_csv("companylist.csv", quotechar='"')
    # tab = pd.read_csv("cac40companyList.csv", quotechar='"')
    for elmt in tab['Symbol']:
        print 'epa%3A'+elmt.split('.')[0],
        try:
            e = predictor(elmt.split('.')[0].upper())
            # e = predictor('EPA%3A'+elmt.split('.')[0].upper())

        except:
            print " FAIL"
            # print traceback.format_exc()
            continue
        print " OK"


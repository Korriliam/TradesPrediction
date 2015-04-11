# from __future__ import unicode_literals
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from StringIO import StringIO
import traceback
import fann_neural_with_orange


__author__ = 'korrigan'

import pandas as pd
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
from os import listdir


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

    UseVolume = False

    """
    Accueille les donnes d'apprentissage.
    """
    TrainData = None

    """
    Stocke les donnes de validation
    """
    ValidationData = None


    ratioUsed = 0.7


    """
    neural network
    """
    nbNode = 25
    regFact = 0.01
    nbnnIt = 1000

    def __init__(self, symbol_of_stock = None, nbj = 60, fileName = "",range_of_prediction=8, training_range = 500):

        if symbol_of_stock is None and file is not "":
            print fileName
            self.classifier = self.LoadClassifier(fileName)
            self.symbol = fileName.split('-')[2]
            tab = pd.read_csv("cac40companyList.csv", quotechar='"')
            for i,elmt in enumerate(tab['Symbol']):
                if elmt == self.symbol:
                    print tab['Name'][i]
            self.bestlearner = fileName.split('-')[1]
            self.nbj = int(fileName.split('-')[3])
        else:
            self.symbol  = symbol_of_stock
            self.nbj = nbj
            self.numberOfDimensionOfDescriptor = 8
            self.Data = dataRecovery.historical_quotes(symbol_of_stock,"20050505","20150408")
            # self.Data = dataRecovery.from_google_historical(symbol_of_stock,"2005-05-05")
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
                Orange.classification.tree.SimpleTreeLearner(min_instances=25),
                Orange.ensemble.forest.RandomForestLearner(),
                Orange.classification.svm.LinearSVMLearner(solver_type=Orange.classification.svm.LinearSVMLearner.L2R_L2LOSS_DUAL,
                                                                   C=1.0,
                                                                   eps=1,
                                                                   normalization=True),
                Orange.classification.knn.kNNLearner(
                                    distance_constructor=Orange.distance.Manhattan(),
                                    k=10
                ),
                fann_neural_with_orange.FannNeuralLearner(),
            ]

            learner = ['nn','naivebaye','majority','simpletree','randomforest','linearsvm','knn','fann']
            self.learner = learner
            self.learners = learners

    def postTreatmentDataset(self, quotes,forReal = False):
        '''
        Va generer les donnes "pen,High,Low,Close,Volume" xx jours, plus le label (si le 6e jour ca monte ou pas)
        :return:
        '''
        # Date Open High Low Close Volume
        variations = []
        size = quotes.shape[0]
        l = range(size-100)
        quotes = quotes[::-1] # on inverse l'ordre
        # oo = np.zeros(ceil(size/10))
        u = 0
        if forReal:
            vec = np.zeros((self.nbj-1)+1)
            quotes = quotes[-self.nbj:]
            j = 0
            rr = self.gainNormaliseParJour(quotes)
            while j != self.nbj-1:
                vec[j] = rr[j]
                j += 1
            vec[self.nbj-1] = 0
            return vec
        else:
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
                    if self.UseVolume:
                        vec[j*5 + 4] = quotes[i+j]['Volume']
                    if not self.UseVolume:
                        j += 1
                    else:
                        j += 2
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

    def print_results(self, res):
        # loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
        accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
        precision = Orange.evaluation.scoring.mlc_precision(res)
        recall = Orange.evaluation.scoring.mlc_recall(res)
        # print 'loss=', loss
        print 'accuracy=', accuracy
        print 'precision=', precision
        print 'recall=', recall
        print

    def runCrossValidation(self):
        try:
            cv = Orange.evaluation.testing.cross_validation(self.learners, self.dataOrange, folds=5)
        except ValueError:
            print self.dataOrange
        accuTab = Orange.evaluation.scoring.CA(cv)
        aucTab = Orange.evaluation.scoring.AUC(cv)


        print "Name     CA        AUC"
        for learner,CA,AUC in zip(self.learners, accuTab, aucTab):
            print "%-8s %.2f      %.2f" % (learner.name, CA, AUC)
        # print "Accuracy:",
        # print ["%.4f" % score for score in accuTab]
        # print "AUC:",
        # print ["%.4f" % score for score in aucTab]
        # print "Ended."
        # self.print_results(cv)

        print "Best AUC : %s by %s" % (max(aucTab),self.learner[aucTab.index(max(aucTab))])
        self.bestauc = max(aucTab)
        self.bestlearner = self.learner[aucTab.index(max(aucTab))]

    def runProportionTest(self,nnn):
        print "pour un ratio de %s " % nnn
        res = Orange.evaluation.testing.proportion_test(self.learners, self.dataOrange, nnn,times=1)
        print "Name     CA        AUC"
        for learner,CA,AUC in zip(self.learners, scoring.CA(res), scoring.AUC(res)):
            print "%-8s %.2f      %.2f" % (learner.name, CA, AUC)

    def TrainBestClassifier(self):
        learner = self.learners[self.learner.index(self.bestlearner)]
        classifier = learner(self.dataOrange)
        self.classifier = classifier

    def SaveClassifier(self,classifier,name):
        with open("Classifier-"+name+"-"+str(self.symbol)+"-"+str(self.nbj)+"-"+str(self.bestauc)+".pkl","wb") as output:
            pickle.dump(classifier,output,pickle.HIGHEST_PROTOCOL)
        print "Classifier saved to %s" % "Classifier-"+name+"-"+str(self.symbol)+"-"+str(self.nbj)+"-"+str(self.bestauc)+".pkl"

    def LoadClassifier(self,fileName):
        try:
            with open(fileName,'rb') as input:
                classifier = pickle.load(input)
        except IOError:
            with open(fileName+".pkl") as input:
                classifier = pickle.load(input)

        return classifier

    def Classify(self,date):
        print date
        if isinstance(date, str):
            eee = dataRecovery.dateutil.parser.parse(date)
            if eee.day-1 < 10 and eee.month < 10:
                strdate = "%s0%s0%s" % (eee.year, eee.month, eee.day-1)
            elif eee.day-1 < 10 and eee.month >= 10:
                strdate = "%s%s0%s" % (eee.year, eee.month, eee.day-1)
            elif eee.day-1 >= 10 and eee.month < 10:
                strdate = "%s0%s%s" % (eee.year, eee.month, eee.day-1)
            else:
                strdate = "%s%s%s" % (eee.year, eee.month, eee.day-1)

        self.Data = dataRecovery.historical_quotes(self.symbol,"20050505",strdate)
        self.defineDomain()
        vec = self.postTreatmentDataset(self.Data, forReal=True)
        rrrrr = vec.tolist()
        rrrrr[self.nbj-1] =  int(rrrrr[rrrrr.__len__()-1])
        try:
            oo = Orange.data.Instance(self.domain, rrrrr)
        except TypeError:
            print(traceback.format_exc())
            pass
        try:
            result = self.classifier(oo)
        except:
            print traceback.format_exc()
            print self.bestlearner
            return "ERROR"
        print result

    def defineDomain(self):
        classattr = Orange.feature.Discrete("class", values=['Up','Down'])
        features = []
        i = 0
        while i != self.nbj-1:
            features.append(Orange.feature.Continuous("NormalizedGain%d" % i))
            # features.append(Orange.feature.Continuous("High%d" % i))
            # features.append(Orange.feature.Continuous("Low%d" % i))
            # features.append(Orange.feature.Continuous("Close%d" % i))
            if self.UseVolume:
                features.append(Orange.feature.Continuous("Volume%d" % i))
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
    classifiersfiles = [elmt for elmt in listdir("/home/korrigan/PycharmProjects/TradesPrediction") if elmt.startswith("Classifier") and not elmt.__contains__("fann")]
    for e in classifiersfiles:
        try:
            e = predictor(fileName=e)
        except:
            # print traceback.format_exc()
            print e
        try:
            e.Classify("20150410")
        except:
            # print traceback.format_exc()
            continue
    # raw_input()
    # # tab = pd.read_csv("companylist.csv", quotechar='"')
    # tab = pd.read_csv("cac40companyList.csv", quotechar='"')
    # with open('res.csv','a+') as output:
    #     for i,elmt in enumerate(tab['Symbol']):
    #         for nb in [700,900,1100,1300,1500,1700]:
    #             print tab['Name'][i]
    #             try:
    #                 e = predictor(symbol_of_stock=elmt,nbj=nb)
    #                 e.runCrossValidation()
    #                 output.write("%s,%s,%s,%s\n" % (tab['Name'][i],e.bestauc,e.bestlearner,e.nbj))
    #                 e.TrainBestClassifier()
    #                 if e.bestauc > 0.7:
    #                     e.SaveClassifier(e.classifier,e.bestlearner)
    #             except:
    #                 print traceback.format_exc()
    #                 continue
    #             print " OK"


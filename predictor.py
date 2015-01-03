__author__ = 'korrigan'

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
import plotly.plotly as py
from plotly.graph_objs import Scatter, Data, Bar, Layout, Figure
from test_polyfit2d import  polyval2d, polyfit2d
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
from nltk.stem import SnowballStemmer
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
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
    Accueille les donn�es d'apprentissage.
    """
    TrainData = None

    """
    Stocke les donn�es de validation
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


        print "Ended."

    def postTreatmentDataset(self, quotes):
        '''
        We will use the addSample function to build our dataset
        :return:
        '''
        e = SupervisedDataSet() # o
        # outp values is True if the stock is increasing this day, of False if it's decreasing.

        variations = []
        for i, quote in enumerate(quotes):
            if i >= 1:
                variation = (quote[0] - quotes[i-1][0])/quotes[i-1][0]
                variations.append(variation)
                gain_volume = (quote[1] - quotes[i-1][1])/quotes[i-1][1]
                gains_volume.append(gain_volume)
            #e.addSample(inp, outp)

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
    e = predictor('GSZ')
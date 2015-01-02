__author__ = 'korrigan'

import cPickle
import numpy as np
from matplotlib.pyplot import figure, show, plot, savefig, legend
import dataRecovery
from datetime import datetime, date, time, timedelta
from math import ceil
from pybrain.datasets import SupervisedDataSet


class predictor(object):

    def __init__(self, symbol_of_stock, range_of_prediction=8, training_range = 500):

        fromDateTrain = datetime.utcnow() - timedelta(days=(training_range+range_of_prediction))#on donne le nb de jour avant auj ou on commence
        toDateTrain = datetime.utcnow() - timedelta(days=range_of_prediction)#on s'arrete tant de j avan auj

        fromDateValidation = datetime.utcnow() - timedelta(days=range_of_prediction)#on donne le nb de jour avant auj ou on commence
        toDateValidation = datetime.utcnow()#on s'arrete tant de j avan auj

        self.numberOfDimensionOfDescriptor = 8

        self.dataRecovery = dataRecovery.dataRecovery()
        self.trainData = self.dataRecovery.yahooDownloader(symbol_of_stock, fromDateTrain, toDateTrain)
        self.ValidationData = self.dataRecovery.yahooDownloader(symbol_of_stock, fromDateValidation, toDateValidation)

        self.postTreatmentDataset()

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
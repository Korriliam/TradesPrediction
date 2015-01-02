#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'korrigan'

import urllib2
from urllib2 import URLError
from datetime import datetime, date, time, timedelta
import  numpy as np
from StringIO import StringIO

class dataRecovery(object):
    '''
    Contains several functions aiming to recover data from various websites
    http://en.wikipedia.org/wiki/List_of_financial_data_feeds
    '''

    def __init__(self):
        pass

    def yahooDownloader(self, stockSymbol, fromDate, toDate):
        '''

        :param stockSymbol
        :param fromDate:
        :param toDate:
        :return:
        '''
        # Converting fromDate & toDate to our different variable

        fromYear = fromDate.year
        fromDay = fromDate.day
        fromMonth = fromDate.month

        toYear = toDate.year
        toDay = toDate.day
        toMonth = toDate.month

        #http://www.jarloo.com/yahoo_finance/
        url = 'http://ichart.yahoo.com/table.csv?s=%(stockSymbol)s&g=%(dayMonthYear)s&a=%(fromMonth)s&b=%(fromDay)s&c=%(fromYear)s&d=%(toMonth)s&e=%(toDay)s&f=%(toYear)s' % {
            'stockSymbol': stockSymbol,
            'dayMonthYear': 'd',
            'fromMonth': fromMonth-1,
            'fromDay': fromDay,
            'fromYear': fromYear,
            'toMonth': toMonth+1,
            'toDay': toDay,
            'toYear': toYear
        }

        try:
            response = urllib2.urlopen(url)
        except URLError:
            print "Pas d'internet ? L'url n'a pas pu etre ouverte"
            exit()

        file = response.read()
        # data conversion to numpy array from csv text file, colonnes: Date	Open High	Low	Close	Volume	Adj Close
        data = np.genfromtxt(StringIO(file), skip_header=1, usecols=(0,1,2,3,4,5), delimiter=',')[::-1]

        return data


    def googleDownloader(self, stockSymbol):
        # Converting fromDate & toDate to our different variable


        #http://www.jarloo.com/yahoo_finance/
        url = 'http://finance.google.com/finance/info?q=%s' % {
            'stockSymbol': stockSymbol,
        }

        try:
            response = urllib2.urlopen(url)
        except URLError:
            print "Pas d'internet ? L'url n'a pas pu etre ouverte"
            exit()

        file = response.read()
        # data conversion to numpy array from csv text file, colonnes: Date	Open High	Low	Close	Volume	Adj Close
        data = np.genfromtxt(StringIO(file), skip_header=1, usecols=(0,1,2,3,4,5), delimiter=',')[::-1]

        return data


if __name__ == '__main__':
    e = dataRecovery()

    d = date(2014, 7, 14)
    t = time(7, 0)
    fromDate = datetime.combine(d,t)

    toDate = datetime.utcnow()


    # fromDate = datetime.utcnow() - timedelta(days=30)#on donne le nb de jour avant auj ou on commence
    # toDate = datetime.utcnow() - timedelta(days=10)#on s'arrete tant de j avan auj

    data = e.yahooDownloader('GSZ', fromDate, toDate)
    pass
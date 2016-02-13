__author__ = 'Aran'

import numpy as np
import pandas as pd
from islrtools import bootstrap as bsp


class Exec09:
    def __init__(self):
        self.df = pd.read_csv("../../ISIRExerciseCode/dataset/Boston.csv")

    def compare_mean(self):
        print np.mean(self.df['medv'])
        ''' The var of mean is var(X)/n, so std is std(X)/sqrt(n) '''
        print np.std(self.df['medv'])/np.sqrt(len(self.df['medv']))
        result = bsp.boot(self.df, Exec09.calculate_mean, 1000)
        print "Bootstrap mean: mean=%s, std=%s" % (result[0], result[1])
        print "The 95% confidence interval is: [", result[0] - 2*result[1], result[0] + 2*result[1], "]"

    def compare_median(self):
        print np.median(self.df['medv'])
        result = bsp.boot(self.df, Exec09.calculate_median, 1000)
        print "Bootstrap median: mean=%s, std=%s" % (result[0], result[1])

    def compare_percentile(self):
        print np.percentile(self.df['medv'], 10)
        result = bsp.boot(self.df, Exec09.calculate_percentile, 1000)
        print "Bootstrap 10 percentile: mean=%s, std=%s" % (result[0], result[1])

    @staticmethod
    def calculate_mean(data, index):
        return np.mean(data.ix[index]['medv'])

    @staticmethod
    def calculate_median(data, index):
        return np.median(data.ix[index]['medv'])

    @staticmethod
    def calculate_percentile(data, index):
        return np.percentile(data.ix[index]['medv'], 10)


if __name__ == '__main__':
    exec09 = Exec09()
    exec09.compare_percentile()

# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics.regressionplots as rp
import lrplot


class Exec15:
    def __init__(self):
        auto = pd.read_csv("../dataset/Boston.csv", index_col=0, na_values=['?'])
        self.df = auto.dropna()
        print self.df.columns

if __name__ == '__main__':
    lr = Exec15()


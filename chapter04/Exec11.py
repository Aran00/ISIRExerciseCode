__author__ = 'ryu'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
from islrtools import lrplot


class Exec11:
    def __init__(self):
        auto = pd.read_csv("../dataset/Boston.csv", index_col=0, na_values=['?'])
        self.df = auto.dropna()
        print self.df.columns
        self.y_col = 'crim'
        self.x_cols = self.df.columns.tolist()
        self.x_cols.remove(self.y_col)
        print self.x_cols

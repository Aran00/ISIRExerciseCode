__author__ = 'ryu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.lda import LDA
from sklearn.qda import QDA
from islrtools import tableplot as tp
from pandas import Series, DataFrame
from sklearn import neighbors


class Exec11:
    def __init__(self):
        auto = pd.read_csv("../dataset/Auto.csv", na_values=['?'])
        self.df = auto.dropna()
        mpg_median = np.median(self.df['mpg'])
        self.df['mpg01'] = self.df['mpg'].map(lambda x: 1 if x > mpg_median else 0)
        del self.df['mpg']
        print self.df.columns
        self.y_col = 'mpg01'
        #self.x_cols = self.df.columns.tolist()
        #self.x_cols.remove(self.y_col)
        #print self.x_cols
        #print self.df['mpg01']

    def scatter_all_data(self):
        #pd.tools.plotting.scatter_matrix(self.df, alpha=0.2)
        #plt.show()
        '''(b) how could we find these most relevant variables? From covariance and scatter plot? Not be so sure about it...'''
        self.x_cols = ['cylinders', 'weight', 'displacement', 'horsepower']

    def divide_data(self):
        self.train_set = self.df.ix[self.df.year % 2 == 0, :]
        self.train_X = self.train_set[self.x_cols]
        self.train_y = self.train_set[self.y_col]
        self.test_set = self.df.ix[self.df.year % 2 != 0, :]
        self.test_X = self.test_set[self.x_cols]
        self.test_y = self.test_set[self.y_col]

    def lda_fit(self):
        lda_res = LDA().fit(self.train_X.values, self.train_y.values)
        pred_y = lda_res.predict(self.test_X.values)
        tp.output_table(pred_y, self.test_y.values)

    def qda_fit(self):
        qda_res = QDA().fit(self.train_X.values, self.train_y.values)
        pred_y = qda_res.predict(self.test_X.values)
        tp.output_table(pred_y, self.test_y.values)

    def logistic_fit(self):
        model = smf.logit("%s~%s" % (self.y_col, "+".join(self.x_cols)), data=self.train_set)
        logistic_res = model.fit()
        prob_y = logistic_res.predict(self.test_X)
        pred_y = Series(prob_y).map(lambda x: 1 if x > 0.5 else 0)
        tp.output_table(pred_y, self.test_y.values)

    def knn_fit(self, n_neighbors):
        weights = 'uniform'
        #weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(self.train_X, self.train_y)
        test_X = self.test_set[self.x_cols].values
        test_y = self.test_set[self.y_col].values
        preds = clf.predict(test_X)
        tp.output_table(preds, test_y)

if __name__ == '__main__':
    exec11 = Exec11()
    exec11.scatter_all_data()
    exec11.divide_data()
    #exec11.lda_fit()
    #exec11.qda_fit()
    #exec11.logistic_fit()
    '''(g) So it tests 1, 10 and 100 and gets that 100 is the best. But how does it choose these 3 params? '''
    exec11.knn_fit(1)

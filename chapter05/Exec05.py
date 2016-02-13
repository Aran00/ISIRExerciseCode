__author__ = 'Aran'

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas import Series
from islrtools import tableplot as tp
from numpy import random as npr


class Exec05:
    def __init__(self):
        self.default = pd.read_csv("../dataset/Default.csv")
        self.x_cols = None
        self.formula = None
        self.y_col = 'default'
        self.reset_x_cols(['income', 'balance'])
        self.default[self.y_col] = self.default[self.y_col].map(lambda x: 1 if x == 'Yes' else 0)
        self.default['student'] = self.default['student'].map(lambda x: 1 if x == 'Yes' else 0)

    def reset_x_cols(self, x_cols):
        self.x_cols = x_cols
        self.formula = "%s~%s" % (self.y_col, "+".join(x_cols))

    def split_train_test(self, train_indexes, use_numpy=True):
        if use_numpy:
            ''' Implementation 1 '''
            test_indexes = np.setdiff1d(self.default.index.values, train_indexes)
            return self.default.ix[train_indexes], self.default.ix[test_indexes]
        else:
            ''' Implementation 2 '''
            train_cond = self.default.index.isin(train_indexes)
            return self.default.ix[train_cond], self.default.ix[~train_cond]

    def logistic_regression(self, train_set, test_set):
        model = smf.logit(self.formula, data=train_set)
        result = model.fit()
        print result.summary()
        probs = Series(result.predict(sm.add_constant(test_set[self.x_cols])))
        pred_values = probs.map(lambda x: 1 if x > 0.5 else 0)
        tp.output_table(pred_values.values, test_set[self.y_col].values)
        print "\n"

    def test_diff_splits(self):
        n = self.default.shape[0]
        train_index = npr.choice(np.arange(n), 0.5*n, replace=False)
        self.logistic_regression(*self.split_train_test(train_index))

if __name__ == '__main__':
    exec05 = Exec05()
    exec05.test_diff_splits()
    exec05.reset_x_cols(['income', 'balance', 'student'])
    exec05.test_diff_splits()
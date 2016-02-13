__author__ = 'Aran'

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import random as npr
from sklearn import linear_model, cross_validation
from islrtools import poly


class Exec08:
    N = 100

    def __init__(self):
        npr.seed(1)
        self.x = npr.normal(0, 1, self.N)
        self.y = self.x - 2 * np.square(self.x) + npr.normal(0, 1, self.N)
        ''' n=100, p=2 '''

    def plot_scatter(self):
        plt.scatter(self.x, self.y,)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def cross_poly_i(self, degree, cv):
        '''
        (d) LOOCV has no relationship with seed
        (e) In most cases the degree 2 has least error, but not always.
        (So we use the first within 1 std ?)
        '''
        clf = linear_model.LinearRegression()
        if degree == 1:
            X = np.vstack(self.x)  #self.df[self.x_cols]
            Z = sm.add_constant(X)
        else:
            Z, norm2, alpha = poly.ortho_poly_fit(self.x, degree)
            X = Z[:, 1:]
        results = sm.OLS(self.y, Z).fit()
        print results.summary()
        ''' These scores should be equal with R for the LOOCV, but not equal for the case folds != data_len '''
        scores = cross_validation.cross_val_score(clf, X, self.y, scoring="mean_squared_error", cv=cv)
        print degree, -np.mean(scores), np.std(scores)


if __name__ == '__main__':
    exec08 = Exec08()
    for i in xrange(4):
        exec08.cross_poly_i(i+1, exec08.N)
    exec08.plot_scatter()

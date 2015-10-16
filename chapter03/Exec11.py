__author__ = 'ryu'

import statsmodels.api as sm
import numpy as np


class Exec11:
    def __init__(self):
        np.random.seed(1)
        self.X = np.random.randn(100)
        '''
        Note that this time of randn returns different consequences,
        That is, the seed function only acts once.
        But note that a different function still use the seed. So if you have the code
            np.random.seed(1)
            x = np.random.randn(5)
            y = np.random.normal(0, 1, 5)
        The x, y would always be the same sequence.
        '''
        self.y = 2 * self.X + np.random.randn(100)
        #print "x=", self.X
        #print "y=", self.y

    def fit_Y(self):
        results = sm.OLS(self.y, self.X).fit()
        print results.summary()
        return results

    def fit_X(self):
        results = sm.OLS(self.X, self.y).fit()
        print results.summary()
        return results


if __name__ == '__main__':
    '''
    (f) We can see the t statistics are the same
    '''
    exec11 = Exec11()
    #exec11.fit_Y()
    #exec11.fit_X()
    exec11.generate_sum_square_equal()
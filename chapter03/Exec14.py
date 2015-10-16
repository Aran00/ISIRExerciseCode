# -*- coding: utf-8 -*- #
__author__ = 'ryu'

import statsmodels.formula.api as smf
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import lrplot as lrp
from pandas import DataFrame


class Exec14:
    def __init__(self):
        np.random.seed(1)
        self.x1 = npr.rand(100)
        self.x2 = 0.5*self.x1 + npr.randn(100)/10
        self.y = 2 + 2*self.x1 + 0.3*self.x2 + npr.randn(100)
        self.df = DataFrame({"x1": self.x1, "x2": self.x2, "y": self.y})

    def cov_and_scatter(self):
        '''
        X = np.vstack((self.x1, self.x2))
        print np.corrcoef(X)
        '''
        print np.corrcoef(self.x1, self.x2)
        plt.scatter(self.x1, self.x2, c='w')
        plt.show()

    def multi_regression(self):
        '''
        (c) In this MLR, the coefficient of x1 and x2 all have large p-value,
            and we can't reject the null hypothesis for β1 or β2.
            But as we can see, F-stat is still large here. So we can reject
               β1=0 && β2=0.
        '''
        res = smf.ols(formula="y ~ x1 + x2", data=self.df).fit()
        print res.summary()
        return res

    def single_regression(self, var_name):
        '''
        Compare this conclusion with figure 3-15 in the book
        (d)(e) We can reject the null hypothesis in SLR.
        (f) No, because x1 and x2 have collinearity, it is hard to distinguish their effects when regressed upon together.
            When they are regressed upon separately, the linear relationship between y and each predictor is indicated more clearly.
        '''
        res = smf.ols(formula="y ~ %s" % var_name, data=self.df).fit()
        print res.summary()
        return res

    def add_extra_point(self):
        '''
        (g) In the first model, it shifts x1 to statistically insignificance,
            and shifts x2 to statistiscal significance from the change in p-values between the two linear regressions.
        '''
        self.df = DataFrame({
            "x1": np.append(self.x1, 0.1),
            "x2": np.append(self.x2, 0.8),
            "y": np.append(self.y, 6)
        })


if __name__ == '__main__':
    exec14 = Exec14()
    #exec14.cov_and_scatter()
    exec14.add_extra_point()
    '''
    (g) From all diagnosis plots,
        the additional point is a high leverage point, but not an outlier(|t_resid|<3).
    '''
    res = exec14.multi_regression()
    lrp.plot_R_graphs(res)
    exec14.single_regression("x1")
    lrp.plot_R_graphs(res)
    exec14.single_regression("x2")
    lrp.plot_R_graphs(res)
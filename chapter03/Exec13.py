__author__ = 'ryu'

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
from pandas import DataFrame


class Exec13:
    def __init__(self, std):
        '''(a) (b) (c)'''
        npr.seed(1)
        self.x = npr.randn(100)
        #print isinstance(self.x, np.ndarray)
        self.eps = npr.normal(0, std, 100)
        #print self.eps
        self.y = -1 + 0.5*self.x + self.eps

    def plot_scatter_and_line(self, result):
        '''(d)(f)'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.x, self.y, c='w')
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        rp.abline_plot(intercept=-1, slope=0.5, ax=ax, c='r', label="model fit")
        rp.abline_plot(model_results=result, ax=ax, c='g', label="pop.regression")
        plt.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()

    def fit_data(self):
        '''
        (e) The linear regression fits a model close to the true value of the coefficients as was constructed.
            The model has a large F-statistic with a near-zero p-value so the null hypothesis can be rejected.
        '''
        #res = sm.OLS(self.y, sm.add_constant(self.x)).fit()
        X = DataFrame({'x': self.x, 'y': self.y})
        res = smf.ols(formula="y ~ x", data=X).fit()
        self.print_regression_result_details(res)
        return res

    def fit_poly_data(self):
        '''
        (g) R^2 only has slight increase, but p-value of I(x**2) is too large to be statistical significant.
        '''
        X = DataFrame({'x': self.x, 'y': self.y})
        res = smf.ols(formula="y ~ x + I(x**2)", data=X).fit()
        self.print_regression_result_details(res)

    def print_regression_result_details(self, results):
        print '\n'
        print results.summary()
        print "The coeffcients are: \n", results.params
        print "The coeffcients intervals are: \n", results.conf_int()
        print "R^2 is ", results.rsquared
        print "RSE is ", np.sqrt(results.mse_resid)


if __name__ == '__main__':
    '''(h)-(j)'''
    exec13 = Exec13(0.25)
    result = exec13.fit_data()
    #exec13.fit_poly_data()
    exec13.plot_scatter_and_line(result)

    exec13 = Exec13(0.125)
    result = exec13.fit_data()
    exec13.plot_scatter_and_line(result)

    exec13 = Exec13(0.5)
    result = exec13.fit_data()
    exec13.plot_scatter_and_line(result)
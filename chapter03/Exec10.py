__author__ = 'ryu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
import lrplot as lrp

class Exec10:
    def __init__(self):
        self.df = pd.read_csv("../dataset/Carseats.csv")

    def multi_variate_regression(self):
        '''
        The answer of exercise-03-10:
        (a) Code
        (b) As Price increases, Sales decreases.
            Urban: high p-value, not statistically significant
            If the store is in the US, the sales will increase by approximately 1201 units.
        (c) Sales = 13.04 + -0.05 Price + -0.02 UrbanYes + 1.20 USYes
        (d) Price and US
        '''
        mod = smf.ols(formula="Sales ~ Price + C(Urban) + C(US)", data=self.df)
        res = mod.fit()
        print res.summary()
        print "RSE of 1st model is ", np.sqrt(res.mse_resid)
        print "\n\n"
        return res

    def multi_smaller_regression(self):
        '''
        The answer of exercise-03-10:
        (e) Code
        (f) Based on the RSE and R^2 of the linear regressions, they both fit the data similarly
            with linear regression from (e) fitting the data slightly better.
            (By what index, F-statistic?)
        (g) Code
        (h) There are.
        '''
        mod = smf.ols(formula="Sales ~ Price + C(US)", data=self.df)
        res = mod.fit()
        print res.summary()
        print "RSE of 2nd model is ", np.sqrt(res.mse_resid)
        print "The coeffcients intervals are: \n", res.conf_int()
        return res

if __name__ == '__main__':
    lr = Exec10()
    #lr.multi_variate_regression()
    res = lr.multi_smaller_regression()
    #rp.influence_plot(res, criterion="DFFITS", size=20)
    lrp.plot_R_graphs(res)
    plt.show()
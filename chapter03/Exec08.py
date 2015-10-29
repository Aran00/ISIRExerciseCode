#!/usr/bin/python
# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics.regressionplots as rp
from islrtools import lrplot


class Exec08:
    def __init__(self):
        auto = pd.read_csv("../dataset/Auto.csv", na_values=['?'])
        self.df = auto.dropna()

    def simple_regession(self):
        ''' The answer of exercise03-08:
        (a)
            (i)  Yes, from F-stat
            (ii) Explain it from RSE and R^2 stat
            (iii)negative
            (iv) Code, no prediction interval
        (b) Code
        (c) Residual/fitted: non-linearity
        '''

        # model = smf.ols(formula="mpg ~ horsepower", data=self.df)
        y = self.df['mpg']
        X = self.df[['horsepower']]
        X = sm.add_constant(X)
        print X
        res = sm.OLS(y, X).fit()
        # res = model.fit()
        print res.summary()

        print "The prediction is: ", res.predict(exog=[[1, 98]])
        print "The prediction interval is: "

        '''
        self.df.plot(kind="scatter", x='horsepower', y='mpg', c='w')
        graph_x = np.linspace(min(self.df['horsepower']), 200)
        graph_y = res.predict(sm.add_constant(graph_x))
        plt.plot(graph_x, graph_y)
        '''
        fig = rp.abline_plot(model_results=res)
        ax = fig.axes[0]
        ax.scatter(X['horsepower'], y, c='w')
        plt.show()
        lrplot.plot_R_graphs(res)

if __name__ == '__main__':
    lr = Exec08()
    lr.simple_regession()

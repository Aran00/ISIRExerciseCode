#!/usr/bin/python
# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics.regressionplots as rp


class Exec08:
    def __init__(self):
        auto = pd.read_csv("../dataset/Auto.csv", na_values=['?'])
        self.df = auto.dropna()

    def simple_regession(self):
        ''' The answer of exercise03-08:
        (a) (i)有 (ii)|t|值很大,p值很小,相关性强 (iii)负相关 (iv)预测区间如何计算?
        (b) 已完成
        (c) Residual/fitted 似乎是一个二次曲线 '''

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


if __name__ == '__main__':
    lr = Exec08()
    lr.simple_regession()

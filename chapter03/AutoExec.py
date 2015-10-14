#!/usr/bin/python
# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import spline

import statsmodels.graphics.regressionplots as rp
from statsmodels.stats.outliers_influence import variance_inflation_factor

class AutoExec:
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

    def plot_scatter_matrix(self):
        pd.tools.plotting.scatter_matrix(self.df, alpha=0.2)
        plt.show()

    def multi_variate_regression(self):
        columns = self.df.columns.values.tolist()
        columns.remove('name')
        X = self.df[columns]
        print np.cov(X, rowvar=0).shape
        cov_df = pd.DataFrame(np.corrcoef(X, rowvar=0), columns=columns, index=columns)
        print "The correlation coefficients of each column is: \n", cov_df

        '''
        The answer of exercise-03-09:
        (a) (b) 略
        (c) (i)有 (ii)displacement, weight, year, origin; year最相关 (iii)说明year与mpg是正相关
        '''
        columns.remove('mpg')
        self.X = sm.add_constant(self.df[columns])
        y = self.df['mpg']
        self.res = sm.OLS(y, self.X).fit()
        print self.res.summary()

        # The Leverage-Studentized Residuals plot
        rp.influence_plot(self.res, criterion="DFFITS", size=20)
        plt.show()

    def plot_predict_residual(self):
        ''' How to fit this type of data? '''
        graph_x = self.res.predict(sm.add_constant(self.X))
        graph_y = self.df['mpg'] - graph_x
        plt.scatter(graph_x, graph_y, c='w')
        '''
        x_new = np.linspace(graph_x.min(), graph_x.max())
        power_smooth = spline(graph_x, graph_y, x_new)
        plt.plot(x_new, power_smooth)
        '''
        plt.show()


if __name__ == '__main__':
    lr = AutoExec()
    #lr.simple_regession()
    lr.multi_variate_regression()
    #lr.plot_predict_residual()
    #lr.test()

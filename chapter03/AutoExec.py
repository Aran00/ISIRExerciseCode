#!/usr/bin/python
# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
from pandas import Series, DataFrame
from scipy.interpolate import spline
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


class Exec09:
    def __init__(self):
        auto = pd.read_csv("../dataset/Auto.csv", na_values=['?'])
        self.df = auto.dropna()
        columns = self.df.columns.tolist()
        columns.remove('name')
        self.df = self.df[columns]
        columns.remove('mpg')
        self.X = sm.add_constant(self.df[columns])

    def plot_scatter_matrix(self):
        '''exercise-03-09(a)'''
        pd.tools.plotting.scatter_matrix(self.df, alpha=0.2)
        plt.show()

    def show_covariance(self):
        '''exercise-03-09(b)'''
        print np.cov(self.df, rowvar=0).shape
        columns = self.df.columns.tolist()
        cov_df = pd.DataFrame(np.corrcoef(self.df, rowvar=0), columns=columns, index=columns)
        print "The correlation coefficients of each column is: \n", cov_df

    def multi_variate_regression(self):
        '''
        exercise-03-09(c)
        (i)有
        (ii)displacement, weight, year, origin; year最相关
        (iii)说明year与mpg是正相关
        '''
        y = self.df['mpg']
        res = sm.OLS(y, self.X).fit()
        print self.res.summary()
        return res

    def plot_graphs(self, res):
        '''
        col_num = self.X.shape[1]
        rp.plot_fit(res, exog_idx=col_num-3)
        rp.plot_regress_exog(res, exog_idx=col_num-1, fig=None)
        rp.plot_leverage_resid2(res)   #squared
        '''
        '''
        The answer of exercise-03-09:
        (d) plot the residual graph and the leverage graph here, and can see outlier and high leverage point here
        '''
        # The fitted value- Studentized residual plot
        self.plot_fittedValue_residual(res)

        # The Leverage-Studentized Residuals plot
        rp.influence_plot(res, criterion="DFFITS", size=20)
        plt.show()

    def get_vifs(self):
        col_num = self.X.shape[1]
        df = self.X.ix[:, 1:]
        vif_list = [variance_inflation_factor(np.array(self.X), i) for i in np.arange(1, col_num, 1)]
        result = Series(vif_list, df.columns)
        print "VIF of all columns are: \n", result

    def get_leverages_resid(self, res):
        infl = res.get_influence()
        leverage = infl.hat_matrix_diag
        resid = infl.resid_studentized_external
        print "leverage is:\n", leverage
        print "studentize residual is:\n", resid
        plt.scatter(leverage, resid, c='w')
        plt.show()

    def plot_fittedValue_residual(self, res):
        infl = res.get_influence()
        resid = infl.resid_studentized_external
        graph_y = resid
        graph_x = res.predict(self.X)
        original_index = self.X.index
        #print graph_x, graph_y
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(graph_x, graph_y, c='w')
        assert(len(graph_y) == len(graph_x))
        print len(graph_y)
        for i in xrange(len(graph_x)):
            if np.abs(graph_y[i]) > 2:
                ax.annotate(original_index[i], (graph_x[i], graph_y[i]), xytext=(-3, 3),
                            textcoords="offset points", size="x-small")
        plt.show()

    def plot_predict_residual(self, res):
        ''' How to fit this type of data? '''
        graph_x = res.predict(self.X)
        graph_y = self.df['mpg'] - graph_x
        plt.scatter(graph_x, graph_y, c='w')
        '''
        x_new = np.linspace(graph_x.min(), graph_x.max())
        power_smooth = spline(graph_x, graph_y, x_new)
        plt.plot(x_new, power_smooth)
        '''
        plt.show()

    def regress_with_interaction(self):
        mod = smf.ols(formula="mpg ~ cylinders * displacement + displacement * weight", data=self.df)
        res = mod.fit()
        print res.summary()

    def regress_with_poly(self):
        pass

if __name__ == '__main__':
    lr = Exec09()
    #lr.show_covariance()
    #lr.simple_regession()
    ''' RegressionResults class '''
    res = lr.multi_variate_regression()
    #lr.plot_predict_residual(res)
    #lr.plot_graphs(res)
    #lr.get_leverages_resid(res)
    #lr.get_vifs()
    #lr.regress_with_interaction()
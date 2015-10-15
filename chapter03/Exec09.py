#!/usr/bin/python
# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
import lrplot as lrp


class Exec09:
    def __init__(self):
        auto = pd.read_csv("../dataset/Auto.csv", na_values=['?'])
        self.df = auto.dropna()
        columns = self.df.columns.tolist()
        columns.remove('name')
        self.df = self.df[columns]

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
        (i) Yes, there is a relationship between the predictors and the response by testing the null hypothesis of whether all the regression coefficients are zero.
            The F-statistic is far from 1 (with a small p-value), indicating evidence against the null hypothesis.
        (ii) Looking at the p-values associated with each predictor’s t-statistic,
            we see that displacement, weight, year, and origin have a statistically significant relationship,
            while cylinders, horsepower, and acceleration do not.
        (iii) The regression coefficient for year, 0.7508, suggests that for every one year, mpg increases by the coefficient.
            In other words, cars become more fuel efficient every year by almost 1 mpg / year.
        '''
        columns = self.df.columns.tolist()
        columns.remove('mpg')
        X = sm.add_constant(self.df[columns])
        y = self.df['mpg']
        res = sm.OLS(y, X).fit()
        print res.summary()
        return res

    def test_graphs(self, res):
        col_num = self.X.shape[1]
        rp.plot_fit(res, exog_idx=col_num-3)
        rp.plot_regress_exog(res, exog_idx=col_num-1, fig=None)
        rp.plot_leverage_resid2(res)   #squared

    def plot_student_residual_leverage(self, res):
        '''
        The answer of exercise-03-09:
        (d) plot the residual graph and the leverage graph here, and can see outlier and high leverage point here
        '''
        # The Leverage-Studentized Residuals plot
        rp.influence_plot(res, criterion="DFFITS", size=20)
        plt.show()

    def regress_with_interaction(self):
        '''
        The answer of exercise-03-09:
        (e) We can see that the 1st pair is not statistically significant while the 2nd pair is.
        '''
        # The problem is: Is the method(choose the pair whose correlation is largest) a common method? What's its background reason?
        mod = smf.ols(formula="mpg ~ cylinders * displacement + displacement * weight", data=self.df)
        res = mod.fit()
        print res.summary()

    def regress_with_poly_1(self):
        '''
        The answer of exercise-03-09:
        (f) 2 problems are observed from the above plots:
            1) the residuals vs fitted plot indicates heteroskedasticity (unconstant variance over mean) in the model.
            2) The Q-Q plot indicates somewhat unnormality of the residuals.
        '''
        # Why choose these predictors? By brute force and choose the least p-value?
        mod = smf.ols(formula="mpg ~ np.log(weight) + np.sqrt(horsepower) + acceleration + I(acceleration**2)", data=self.df)
        res = mod.fit()
        print res.summary()
        return res

    def regress_with_poly_2(self):
        '''
        The answer of exercise-03-09:
        (f) From the correlation matrix in 9a., displacement, horsepower and weight show a similar nonlinear pattern against our response mpg.
            This nonlinear pattern is very close to a log form.
            So in the next attempt, we use log(mpg) as our response variable.
        '''
        # Why choose these predictors? By brute force and choose the least p-value?
        mod = smf.ols(formula="np.log(mpg) ~ cylinders+displacement+horsepower+weight+acceleration+year+origin", data=self.df)
        res = mod.fit()
        print res.summary()
        return res


if __name__ == '__main__':
    ex09 = Exec09()
    #ex09.plot_scatter_matrix()
    #ex09.show_covariance()
    ''' RegressionResults class '''
    res = ex09.multi_variate_regression()
    #lrp.plot_scale_location(res)
    #lrp.plot_qq(res)
    lrp.plot_fitted_student_residual(ex09.df, res)

    #ex09.get_leverages_resid(res)
    #ex09.get_vifs(ex09.X)
    #ex09.regress_with_interaction()

    #res = ex09.regress_with_poly_2()
    #lrp.plot_graphs_like_R(res)
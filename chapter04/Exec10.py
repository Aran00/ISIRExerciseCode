__author__ = 'ryu'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
from pandas import DataFrame, Series
from islrtools import tableplot as tp
from islrtools import calcustat as cs

class Exec10:
    def __init__(self):
        auto = pd.read_csv("../dataset/weekly.csv", index_col=0)
        self.df = auto.dropna()
        #print self.df.columns
        self.y_col = 'Direction'
        self.x_cols = self.df.columns.tolist()
        self.x_cols.remove(self.y_col)
        #print self.x_cols
        self.transformedDF = DataFrame(self.df[self.x_cols])
        '''Use the index in model.endog_names here'''
        self.transformedDF[self.y_col] = self.df[self.y_col].map(lambda x: 1 if x == 'Up' else 0)

    def search_pattern(self):
        '''
        (a) Seems year-volume has a quad relationship, and their covariance is quite large.
        '''
        cs.get_covariance(self.transformedDF)
        #print self.transformedDF.ix[1, :]
        pd.tools.plotting.scatter_matrix(self.transformedDF, alpha=0.2)
        plt.show()

    def logistic_regression(self):
        '''
        (b) it seems the statistical significant predict variable is only Lag2. How disappointing...
        '''
        formula = "Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume"
        #model = smf.glm(formula, data=self.df, family=sm.families.Binomial())
        model = smf.logit(formula, data=self.transformedDF)
        result = model.fit()
        print result.summary()
        #print result.fittedvalues
        '''Beware the prob here is the index 0's prob, so we should use the lambda function below'''
        #pred_values = result.fittedvalues.map(lambda x: 0 if x > 0.5 else 1)
        probs = result.predict(sm.add_constant(self.df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]))
        #print pred_values
        '''
        (c) Percentage of currect predictions: (54+557)/(54+557+48+430) = 56.1%.
            Weeks the market goes up the logistic regression is right most of the time, 557/(557+48) = 92.1%.
            Weeks the market goes up the logistic regression is wrong most of the time 54/(430+54) = 11.2%.
        '''
        #tp.output_table(pred_values.values, self.transformedDF[self.y_col].values)




if __name__ == '__main__':
    exec10 = Exec10()
    #exec10.search_pattern()
    exec10.logistic_regression()

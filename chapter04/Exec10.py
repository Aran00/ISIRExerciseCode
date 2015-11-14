__author__ = 'ryu'

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as rp
from pandas import DataFrame, Series
from islrtools import tableplot as tp
from islrtools import calcustat as cs
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import neighbors


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

    def logistic_regression(self, use_glm=True):
        '''
        (b) it seems the statistical significant predict variable is only Lag2. How disappointing...
        '''
        formula = "Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume"
        model = smf.glm(formula, data=self.df, family=sm.families.Binomial()) if use_glm else smf.logit(formula, data=self.transformedDF)
        result = model.fit()
        if use_glm:
            probs = result.fittedvalues
            '''Beware the prob here is the index 0's prob, so we should use the lambda function below'''
            pred_values = probs.map(lambda x: 0 if x > 0.5 else 1)
        else:
            '''The probability of being 1'''
            probs = Series(result.predict(sm.add_constant(self.df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']])))
            pred_values = probs.map(lambda x: 1 if x > 0.5 else 0)
        '''
        (c) Percentage of currect predictions: (54+557)/(54+557+48+430) = 56.1%.
            Weeks the market goes up the logistic regression is right most of the time, 557/(557+48) = 92.1%.
            Weeks the market goes up the logistic regression is wrong most of the time 54/(430+54) = 11.2%.
        '''
        tp.output_table(pred_values.values, self.transformedDF[self.y_col].values)

    def test_all_methods(self):
        x_cols = ['Lag2']
        formula = "Direction~Lag2"
        #print self.df.shape[0]
        train_data = self.df.ix[(self.df['Year'] >= 1990) & (self.df['Year'] <= 2008), :]
        #print train_data.shape[0]
        ''' (d) logistic'''
        model = smf.glm(formula, data=train_data, family=sm.families.Binomial())
        result = model.fit()
        test_data = self.df.ix[self.df['Year'] > 2008, :]
        probs = Series(result.predict(sm.add_constant(test_data[['Lag2']])))
        pred_values = probs.map(lambda x: "Down" if x > 0.5 else "Up")
        tp.output_table(pred_values.values, test_data[self.y_col].values)

        train_X = train_data[x_cols].values
        train_y = train_data[self.y_col].values
        test_X = test_data[x_cols].values
        test_y = test_data[self.y_col].values
        ''' (e) LDA '''
        lda_res = LDA().fit(train_X, train_y)
        pred_y = lda_res.predict(test_X)
        tp.output_table(pred_y, test_y)
        ''' (f) QDA '''
        qda_res = QDA().fit(train_X, train_y)
        pred_y = qda_res.predict(test_X)
        tp.output_table(pred_y, test_y)
        ''' (g) KNN '''
        clf = neighbors.KNeighborsClassifier(1, weights="uniform")
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        tp.output_table(pred_y, test_y)
        ''' (h) logistic and LDA '''
        ''' (i) Is the purpose of the last question going through all methods with no direction?'''

if __name__ == '__main__':
    exec10 = Exec10()
    #exec10.search_pattern()
    exec10.test_all_methods()

# -*- coding: utf-8 -*- #
__author__ = 'Aran'

import pandas as pd
import statsmodels.formula.api as smf


class Exec15:
    def __init__(self):
        auto = pd.read_csv("../dataset/Boston.csv", index_col=0, na_values=['?'])
        self.df = auto.dropna()
        print self.df.columns
        self.y_col = 'crim'
        self.x_cols = self.df.columns.tolist()
        self.x_cols.remove(self.y_col)
        print self.x_cols

    def single_regressions(self):
        '''
        (a) Here
        (b)(c) Omitted
        '''
        for i in xrange(len(self.x_cols)):
            res = smf.ols(formula="%s~%s" % (self.y_col, self.x_cols[i]), data=self.df).fit()
            ''' pvalues is a Series object, and its length is p+1'''
            print "The p-value of %s fit is: " % self.x_cols[i], res.pvalues[1]
            if res.pvalues[1] > 0.05:
                print self.x_cols[i], "doesn't have statistical significant relation with", self.y_col

    def poly_regressions(self):
        ''' (d) Output all the ss poly items '''
        for i in xrange(len(self.x_cols)):
            '''
            Note:
            The poly in R relates with Orthogonal polynomial, and is not the same as the formula below.
            So the program would not have the same answer.
            However it doesn't matter now
            '''
            formula = "{0}~{1}+I({1}**2)+I({1}**3)".format(self.y_col, self.x_cols[i])
            #formula = "{0}~Poly({1},3)".format(self.y_col, self.x_cols[i])
            print formula
            res = smf.ols(formula=formula, data=self.df).fit()
            print res.summary()
            ss_list = []
            for j in xrange(1, 4):
                if res.pvalues[j] < 0.05:
                    ss_list.append(str(j))
            print self.x_cols[i], "#", ",".join(ss_list)


if __name__ == '__main__':
    exec15 = Exec15()
    exec15.poly_regressions()


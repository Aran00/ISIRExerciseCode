__author__ = 'Aran'

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from islrtools import bootstrap as bsp


class Exec06:
    x_cols = ['income', 'balance']
    y_col = 'default'
    formula = "%s~%s" % (y_col, "+".join(x_cols))

    def __init__(self):
        self.default = pd.read_csv("../dataset/Default.csv")
        # self.default[self.y_col] = self.default[self.y_col].map(lambda x: 1 if x == 'Yes' else 0)

    def print_summary(self):
        model = smf.glm(Exec06.formula, data=self.default, family=sm.families.Binomial())
        result = model.fit()
        print result.summary()

    @staticmethod
    def boot_fn(data, index):
        # model = smf.logit(Exec06.formula, data=data.ix[index])
        model = smf.glm(Exec06.formula, data=data.ix[index], family=sm.families.Binomial())
        result = model.fit()
        return result.params['Intercept'], result.params['income'], result.params['balance']


if __name__ == '__main__':
    exec06 = Exec06()
    print bsp.boot(exec06.default, Exec06.boot_fn, 50)
    exec06.print_summary()

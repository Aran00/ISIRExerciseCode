__author__ = 'ryu'

import pandas as pd
import matplotlib.pyplot as plt

class CollegeExec:
    def __init__(self):
        self.college = pd.read_csv("../dataset/College.csv", index_col=0)

    def read_csv_a_b(self):
        '''
        college = pd.read_csv("../dataset/College.csv")     #a
        series = college.ix[:, 0]
        self.college = college.drop('Unnamed: 0', axis=1)   #b
        #print self.college.shape
        '''
        print self.college.shape
        #print self.college

    def plot_data_c_iii(self):
        #axes = pd.tools.plotting.scatter_matrix(self.college.ix[:, 0:10], alpha=0.2)
        self.college.boxplot(column='Outstate', by='Private')
        plt.show()

    def plot_data_c_iv(self):
        college = self.college
        college['Elite'] = college['Top10perc'].map(lambda x: "Yes" if x > 50 else "No")
        college.boxplot(column='Outstate', by='Elite')
        plt.show()

    def plot_data_c_v(self):
        top10Series = self.college['Top10perc']
        plt.figure(1)
        plt.subplot(221)
        top10Series.plot(kind='hist', bins=4, color='w')

        plt.subplot(222)
        top10Series.plot(kind='hist', bins=6, color='w')

        plt.subplot(223)
        top10Series.plot(kind='hist', bins=8, color='w')

        plt.subplot(224)
        top10Series.plot(kind='hist', bins=10, color='w')

        plt.show()

if __name__ == '__main__':
    ce = CollegeExec()
    ce.plot_data_c_v()


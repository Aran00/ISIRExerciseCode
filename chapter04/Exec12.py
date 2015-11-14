__author__ = 'ryu'

import numpy as np
import matplotlib.pyplot as plt

class Exec12:
    def __init__(self):
        pass

    def power(self):
        print np.power(2, 3)

    def power2(self, x, a):
        print np.power(x, a)

    def power3(self, x, a):
        return np.power(x, a)

    def plot_log_graph(self):
        x = range(1, 11)
        y = self.power3(x, 2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e0, 1e1)
        ax.set_ylim(1e0, 1e2)
        #plt.plot(x, y)
        ax.scatter(x, y)
        plt.show()

    def plot_power_graph(self, x, a):
        y = self.power3(x, a)
        #plt.plot(x, y)
        plt.scatter(x, y)
        plt.show()


if __name__ == '__main__':
    exec12 = Exec12()
    #exec12.plot_log_graph()
    exec12.plot_power_graph(range(1, 11), 3)

__author__ = 'Aran'

import pandas as pd
import matplotlib.pyplot as plt


class AutoExec:
    def __init__(self):
        auto = pd.read_csv("../dataset/Auto.csv")
        self.auto = auto.dropna()





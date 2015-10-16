__author__ = 'ryu'

import numpy as np
import random
import rpy2.robjects as robjects


class Exec12:
    def __init__(self):
        pass

    '''Would not be used in my other programs'''
    def generate_same_random_like_R(self):
        '''
        Link: http://stackoverflow.com/questions/22213298/creating-same-random-number-sequence-in-python-numpy-and-r,
        It seems the same seed in python, numpy and R doesn't generate the same random number.
        So a complete same sequence can be only got by calling R by python like this:
        '''
        data = robjects.r("""
            set.seed(1)
            x <- rnorm(100)
        """)
        print data

    def generate_sum_square_equal(self):
        '''Exec12, not the exact same number as in R, reason as above'''
        np.random.seed(1)
        x = np.random.randn(100)
        y = - np.array(random.sample(x, 100))
        print "x=", x
        print "y=", y

        '''np.func can use list, ndarray and Series as params'''
        print (np.sum(x**2))
        print (np.sum(y**2))


if __name__ == '__main__':
    '''
    (f) We can see the t statistics are the same
    '''
    exec12 = Exec12()
    #exec12.generate_same_random_like_R()
    exec12.generate_sum_square_equal()
__author__ = 'Aran'

import numpy as np

'''
Not too much, the only interesting thing is how to get a list without one element.
Can set like this:
'''

list_a = range(4)
print list_a
for i in xrange(4):
    list_b = list_a[:i] + list_a[(i+1):]
    print list_b

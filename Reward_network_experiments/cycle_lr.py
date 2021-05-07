#import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'

def f(n,step_size,y_min,y_max):
    n     = n%(2*step_size)
    delta = (y_max-y_min)/step_size
    if n<step_size:
        return y_min+delta*n
    else:
        return y_max-delta*(n-step_size)


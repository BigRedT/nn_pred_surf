import numpy as np


def asincx_bcosdx(
        a,
        b,
        c,
        d,
        data):
    x = data[:,0]
    y = data[:,1]
    y_ = a*np.sin(c*x)+b*np.cos(d*x)
    return np.sign(y-y_)
    

def ax_b(a,b,data):
    x = data[:,0]
    y = data[:,1]
    y_ = a*x + b
    return np.sign(y-y_)

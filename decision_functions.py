import numpy as np


def asincx_bcosdx(
        a,
        b,
        c,
        d,
        data):
    # pred = sign{y - (asin(cx) + bcos(dx))}
    x = data[:,0]
    y = data[:,1]
    y_ = a*np.sin(c*x)+b*np.cos(d*x)
    return np.sign(y-y_)
    

def ax_b(a,b,data):
    # pred = sign{y - (ax + b)}
    x = data[:,0]
    y = data[:,1]
    y_ = a*x + b
    return np.sign(y-y_)


def absx(data):
    # pred = sign{y - |x|}
    x = data[:,0]
    y = data[:,1]
    y_ = np.abs(x)
    return np.sign(y-y_)


def ellipse(a,b,data):
    # pred = sign{1 - ((x/a)^2 + (y/b)^2)}
    x = data[:,0]
    y = data[:,1]
    return np.sign(1 - (x/a)**2 - (y/b)**2)

import decision_functions
from data import sample

import numpy as np


def test_asincx_bcosdx():
    num_samples = 10
    domain = [(-1,1),(-1,1)]
    data = sample.sample_training_data(
        num_samples,
        domain)

    a = 1
    b = 0
    c = np.pi
    d = np.pi
    labels = decision_functions.asincx_bcosdx(
        a,b,c,d,data)
    
    print(np.hstack((data, np.expand_dims(labels,axis=1))))


def test_ax_b():
    num_samples = 10
    domain = [(-1,1),(-1,1)]
    data = sample.sample_training_data(
        num_samples,
        domain)

    a = 0
    b = 0
    labels = decision_functions.ax_b(
        a,b,data)
    
    print(np.hstack((data, np.expand_dims(labels,axis=1))))
    

def main():
    test_asincx_bcosdx()
    test_ax_b()
    

if __name__=='__main__':
    main()
        

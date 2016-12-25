from visualizers import surface_plot
from data import sample
import decision_functions

import numpy as np
import matplotlib.pyplot as plt


def test_pred_surf_plot():
    domain = [(-2,2),(-2,2)]
    steps = [0.01,0.01]
    meshgrid, data = sample.sample_grid_data(
        domain, steps, reshape=True)

    a = 0.3
    b = 0.7
    c = np.pi
    d = np.pi/2.
    labels = decision_functions.asincx_bcosdx(
        a,b,c,d,data)
    
    fig = surface_plot.pred_surf_plot(
        meshgrid,
        (labels+1)*0.5,
        domain,
        pred_format='array',
        title='Ground truth')
    
    plt.savefig(
        'test_pred_surf_plot.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches = 0.3)
    
    
def main():
    test_pred_surf_plot()

    
if __name__=='__main__':
    main()

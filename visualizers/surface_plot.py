from data import sample 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def pred_surf_plot(
        meshgrid,
        pred,
        domain,
        rstride=1,
        cstride=1,
        pred_format='grid',
        title=None):

    assert(len(meshgrid)==2), 'expects grid to be a list of length 2'
    assert(pred_format in {'grid','array'}), 'pred_format can either be grid or array'

    if pred_format=='array':
        pred = sample.array2grid(pred,meshgrid[0].shape)
     
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot_surface(
        meshgrid[0],
        meshgrid[1],
        pred,
        rstride=rstride,
        cstride=cstride,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True)
    
    cset = ax.contourf(
        meshgrid[0],
        meshgrid[1],
        pred,
        zdir='z',
        offset=-2,
        cmap=cm.coolwarm)

    ax.set_xlabel('x',fontweight='bold')
    ax.set_xlim(domain[0][0], domain[0][1])
    ax.set_ylabel('y',fontweight='bold')
    ax.set_ylim(domain[1][0], domain[1][1])
    ax.set_zlabel('pred',fontweight='bold',labelpad=10.)
    ax.set_zlim(-2, 1.)

    if title:
        ax.set_title(title, y=1.08, fontweight='bold')
        
    return fig

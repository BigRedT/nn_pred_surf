from visualizers import surface_plot
import data.sample
import graph
import train

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os

def get_prediction_surface(
        network,
        sess,
        domain,
        steps):
    
    meshgrid, grid_data = data.sample.sample_grid_data(
        domain,
        steps,
        reshape=True)

    inputs = {'data' : grid_data, 'is_training' : False}
    feed_dict = network.create_feed_dict(inputs)
    prob = sess.run(network.inference.get_prob(),feed_dict=feed_dict)

    fig = surface_plot.pred_surf_plot(
        meshgrid,
        prob[:,0],
        domain,
        rstride=5,
        cstride=5,
        pred_format='array')

    return fig


def get_gt_surface(
        func,
        domain,
        steps):

    meshgrid, grid_data = data.sample.sample_grid_data(
        domain,
        steps,
        reshape=True)

    
    prob_1 = 0.5*(func(grid_data)+1.)

    fig = surface_plot.pred_surf_plot(
        meshgrid,
        prob_1,
        domain,
        rstride=5,
        cstride=5,
        pred_format='array')

    return fig

    
def run(c):
    train_data = data.sample.sample_training_data(
        c.num_train_samples,
        c.domain,
        seed=0)

    network = graph.Graph(
        c.input_dims,
        c.hidden_units,
        c.activation,
        c.keep_prob,
        c.use_batchnorm,
        c.learning_rate)

    sess = tf.Session(graph=network.tf_graph)

    labels = c.decision_func(train_data)

    train.train_model(
        network,
        sess,
        train_data,
        labels,
        c.batch_size,
        c.num_epochs,
        seed=1)

    experiments_dir = c.get_experiments_dir()

    pred_filepath = os.path.join(
        experiments_dir,
        c.pred_filename)

    gt_filepath = os.path.join(
        experiments_dir,
        c.gt_filename)

    fig = get_prediction_surface(
        network,
        sess,
        c.domain,
        c.steps)
    
    plt.savefig(
        pred_filepath,
        dpi=300,
        bbox_inches='tight',
        pad_inches = 0.3)

    fig = get_gt_surface(
        c.decision_func,
        c.domain,
        c.steps)

    plt.savefig(
        gt_filepath,
        dpi=300,
        bbox_inches='tight',
        pad_inches = 0.3)
    

    sess.close()

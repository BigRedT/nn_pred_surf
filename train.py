from visualizers import surface_plot
from data.batch_generator import create_batch_generator
import data.sample
import decision_functions
import graph

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def print_dict(dictionary):
    sorted_keys = dictionary.keys()
    sorted_keys.sort()
    for key in sorted_keys:
        print('{} : {}'.format(key, dictionary[key]))
        

def train_model(
        network,
        sess,
        data,
        labels,
        batch_size,
        num_epochs,
        seed=None):
        
    batches = create_batch_generator(
        data,
        batch_size,
        num_epochs,
        labels,
        seed)

    net_vars = {
        'train_op' : network.get_train_op(),
        'cross_entropy_loss' : network.get_loss('cross_entropy'),
        'regularization_loss' : network.get_loss('regularization'),
        'total_loss' :  network.get_loss('total'),
        'prob' : network.inference.get_prob(),
    }

    vars_to_eval = {
        'cross_entropy_loss' : net_vars['cross_entropy_loss'],
        'regularization_loss:' : net_vars['regularization_loss'],
        'total_loss' : net_vars['total_loss'],
        'prob' : net_vars['prob'],
    }

    network.initialize(sess)
    
    for iter, batch in enumerate(batches):
        print('-'*10)
        print('Iter: ' + str(iter))

        inputs = {
            'data': batch['data'],
            'labels': batch['labels']
        }

        feed_dict  = network.create_feed_dict(inputs)
        
        sess.run(net_vars['train_op'], feed_dict=feed_dict)

        eval_vars = network.eval_vars(vars_to_eval,feed_dict,sess)

        print(eval_vars['total_loss'])
        print(np.max(eval_vars['prob']))
        print(np.min(eval_vars['prob']))

    
def get_prediction_surface(
        network,
        sess,
        domain,
        steps):
    
    meshgrid, grid_data = data.sample.sample_grid_data(
        domain,
        steps,
        reshape=True)

    inputs = {'data' : grid_data}
    feed_dict = network.create_feed_dict(inputs)
    prob = sess.run(network.inference.get_prob(),feed_dict=feed_dict)

    fig = surface_plot.pred_surf_plot(
        meshgrid,
        prob[:,0],
        domain,
        rstride=5,
        cstride=5,
        pred_format='array')

    plt.show()


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

    plt.show()
    
def main():
    input_dims = 2
    hidden_units = [10,10,10,10,10,10,10]
    activation = tf.nn.relu
    keep_prob = 1.
    use_batchnorm = False
    is_training = True
    learning_rate = 0.01
    batch_size = 1000
    num_epochs = 100
    num_train_samples = 10000
    domain = [(-2,2),(-2,2)]
    steps = [0.01,0.01]
    
    network = graph.Graph(
        input_dims,
        hidden_units,
        activation,
        keep_prob,
        use_batchnorm,
        is_training,
        learning_rate)

    sess = tf.Session(graph=network.tf_graph)
    
    train_data = data.sample.sample_training_data(
        num_train_samples,
        domain,
        seed=0)

    decision_func = lambda x: decision_functions.asincx_bcosdx(
        1,
        0,
        np.pi,
        np.pi,
        x) 
    
    labels = decision_func(train_data)
    # decision_functions.asincx_bcosdx(
        # 1,
        # 0,
        # np.pi,
        # np.pi,
        # train_data)
    
    train_model(
        network,
        sess,
        train_data,
        labels,
        batch_size,
        num_epochs,
        seed=None)
    
    get_prediction_surface(
        network,
        sess,
        domain,
        steps)

    get_gt_surface(
        decision_func,
        domain,
        steps)
    
    sess.close()
    
if __name__=='__main__':
    main()
    

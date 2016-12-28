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
            'labels': batch['labels'],
            'is_training': True,
        }

        feed_dict  = network.create_feed_dict(inputs)
        
        sess.run(net_vars['train_op'], feed_dict=feed_dict)

        eval_vars = network.eval_vars(vars_to_eval,feed_dict,sess)

        print(eval_vars['total_loss'])

    
    

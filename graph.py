from inference import Inference

import tensorflow as tf
import numpy as np
from pyAIUtils.aiutils.tftools import placeholder_management
import pdb


class Graph():
    def __init__(
            self,
            input_dims,
            hidden_units,
            activation,
            keep_prob,
            use_batchnorm,
            learning_rate,
            seed=1):
        # Args:
        #   - input_dims : The dimension of a single sample
        #   - hidden_units : A list containing the number of units in each hidden layer
        #   - activation : Activation function to use such as tf.nn.relu
        #   - keep_prob : Keep probability to be used for dropout
        #   - use_batchnorm : If True, batch normalization is applied to each hidden layer
        #   - learning_rate : learning rate to be used if is_training is True
        
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            tf.set_random_seed(seed)
            self.plh = self.create_placeholders(input_dims)
            
            with tf.variable_scope('inference'):
                self.inference = Inference(
                    self.plh['data'],
                    hidden_units,
                    activation,
                    keep_prob,
                    use_batchnorm,
                    self.plh['is_training'])

            with tf.variable_scope('loss'):
                logits = self.inference.get_logits()
                self.losses = self.compute_losses(
                    logits,
                    self.plh['labels'])
                self.total_loss = self.losses['cross_entropy'] + 1e-4*self.losses['regularization']
                
            with tf.variable_scope('optimizer'):
                self.train_op = self.attach_optimizer(self.total_loss,learning_rate)

            self.init = tf.initialize_all_variables()
            
    def create_placeholders(self,input_dims):
        plh = placeholder_management.PlaceholderManager()

        plh.add_placeholder(
            'data',
            tf.float32,
            [None,input_dims])

        plh.add_placeholder(
            'labels',
            tf.float32,
            [None])

        plh.add_placeholder(
            'is_training',
            tf.bool,
            [])

        return plh

    def compute_losses(self,logits,labels):
        losses = dict()

        losses['cross_entropy'] = self.cross_entropy_loss(logits,labels)
        # losses['cross_entropy'] = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits,
        #     targets)

        vars_to_regularize = self.tf_graph.get_collection('to_regularize')
        losses['regularization'] = 0.
        for var in vars_to_regularize:
            losses['regularization'] += tf.nn.l2_loss(var)

        return losses

    def cross_entropy_loss(self, logits, labels):
        targets = 0.5*(labels+1.)
        prob_1, prob_0 = tf.unstack(self.inference.get_prob(),num=2,axis=1)
        epsilon = 1e-2
        loss = -1*(targets*tf.log(prob_1+epsilon) + (1-targets)*tf.log(prob_0+epsilon))
        loss = 2*tf.reduce_mean(loss)

        return loss
    
    def attach_optimizer(self, loss, learning_rate):
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = opt.minimize(loss)
        return train_op
            
    def create_feed_dict(self, inputs):
        feed_dict = dict()
        for plh_name, plh_value in inputs.items():
            feed_dict[self.plh[plh_name]] = inputs[plh_name]

        return feed_dict

    def get_train_op(self):
        return self.train_op

    def eval_vars(self,vars_to_eval,feed_dict,sess):
        var_names = vars_to_eval.keys()
        vars_list = [vars_to_eval[name] for name in var_names]
        eval_vars_list = sess.run(vars_list,feed_dict=feed_dict)
        eval_vars = {k : v for (k,v) in zip(var_names,eval_vars_list)}
        return eval_vars

    def get_loss(self,loss_type):
        if loss_type=='cross_entropy':
            return self.losses['cross_entropy']
        elif loss_type=='regularization':
            return self.losses['regularization']
        elif loss_type=='total':
            return self.total_loss
        else:
            assert(False), 'loss_type not found'

    def initialize(self,sess):
        sess.run(self.init)
            

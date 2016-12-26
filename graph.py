from inference import Inference

import tensorflow as tf
import numpy as np
import pyAIUtils.aiutils.tftools as tftools

class Graph():
    def __init__(
            self,
            input_dims,
            hidden_units,
            activation,
            keep_prob,
            use_batchnorm,
            is_training,
            learning_rate):
        # Args:
        #   - input_dims : The dimension of a single sample
        #   - hidden_units : A list containing the number of units in each hidden layer
        #   - activation : Activation function to use such as tf.nn.relu
        #   - keep_prob : Keep probability to be used for dropout
        #   - use_batchnorm : If True, batch normalization is applied to each hidden layer
        #   - is_training : Mode distinguishing training vs test
        #   - learning_rate : learning rate to be used if is_training is True
        
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.plh = self.create_placeholders(input_dims)

            with tf.variable_scope('inference'):
            self.inference = Inference(
                self.plh['data'],
                hidden_units,
                activation,
                keep_prob,
                use_batchnorm,
                is_training)

            with tf.variable_scope('loss'):
                logits = self.inference.get_logits()
                self.losses = self.compute_losses(
                    logits,
                    self.plh['labels'])
                self.total_loss = self.losses['cross_entropy'] + self.losses['regularization']
                
            if is_training:
                with tf.variable_scope('optimizer'):
                    self.attach_optimizer(self.total_loss)

    def create_placeholders(self,input_dims):
        plh = tftools.placeholder_management.PlaceholderManager()
        plh.add_placeholder(
            'data',
            tf.float32,
            [None,input_dims])

        plh.add_placeholder(
            'labels',
            tf.float32,
            [None])

    def compute_losses(self,logits,labels):
        losses = dict()

        targets = 0.5*(labels+1.)
        losses['cross_entropy'] = tf.nn.sigmoid_cross_entropy_with_logits(
            logits,
            targets)

        vars_to_regularize = self.tf_graph.get_collection('to_regularize')
        losses['regularization'] = 0.
        for var in vars_to_regularize:
            losses['regularization'] += tf.nn.l2_loss(var)

        return losses

    def attach_optimizer(self, loss, learning_rate):
        opt = tf.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = opt.minimize(loss)
            
    def create_feed_dict(self, inputs):
        feed_dict = dict()
        for plh_name, plh_value in inputs.items():
            feed_dict[self.plh[plh_name]] = plh_value

        return feed_dict

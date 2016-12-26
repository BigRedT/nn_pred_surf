import numpy as np
import tensorflow as tf
import pyAIUtils.aiutils.tftools as tftools 

class Inference():
    def __init__(
            self,
            data,
            hidden_units,
            activation,
            keep_prob,
            use_batchnorm,
            is_training):
        # Args:
        #   - data : num_samples x dimensions tensorflow variable
        #   - hidden_units : A list containing the number of units in each hidden layer
        #   - activation : Activation function to use such as tf.nn.relu
        #   - keep_prob : Keep probability to be used for dropout
        #   - use_batchnorm : If True, batch normalization is applied to each hidden layer
        #   - is_training : Mode distinguishing inference for training vs test

        assert(not(use_batchnorm and keep_prob<1.)), 'use either batchnorm or dropout'

        self.activation = activation
        self.keep_prob = keep_prob
        self.use_batchnorm = use_batchnorm
        self.is_training = is_training
        
        num_hidden_layers = len(hidden_units)

        self.hidden_layers = [None]*num_hidden_layers
        
        with tf.variable_scope('hidden_layer_0'):
            self.hidden_layers[0] = self.create_hidden_layer(
                data,
                hidden_units[0])
            
        for i in range(1,num_hidden_layers):
            with tf.variable_scope('hidden_layer_' + str(i)):
                self.hidden_layers[i] = self.create_hidden_layer(
                    self.hidden_layers[i-1],
                    hidden_units[i])
                
        with tf.variable_scope('output_layer'):
            self.output_layer = tftools.layers.full(
                self.hidden_layers[-1],
                2,
                func=None)

            self.prob = tf.nn.softmax(self.output_layer)

    def create_hidden_layer(self, input, num_hidden_units):
        hidden_layer = tftools.layers.full(
            input,
            num_hidden_units,
            func=None)
        
        if self.use_batchnorm:
            hidden_layer = tftools.layers.batch_norm(
                hidden_layer,
                tf.constant(self.is_training))

        hidden_layer = self.activation(hidden_layer)
        hidden_layer = tf.nn.dropout(hidden_layer,self.keep_prob)

        return hidden_layer
        
    def get_logits(self):
        return self.output_layer

    def get_prob(self):
        return self.prob

    def get_hidden_layers(self):
        return self.hidden_layers

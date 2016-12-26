from pyAIUtils.aiutils.data import batch_creators

import numpy as np


def create_batch_generator(data,batch_size,num_epochs,labels=None,seed=None):
    # Args:
    #   - data : num_samples x dimensions ndarray
    #   - batch_size : Size of mini-batch
    #   - num_epochs : Number of epochs
    #   - labels : num_samples dimensional array containing data labels
    #   - seed : Seed for the random number generator

    # Returns:
    #   - batch : A generator which yields a dict with keys 'data' and 'labels'
    #       and their corresponding values
          
    if seed is not None:
        np.random.seed(seed)
        
    num_samples = data.shape[0]
    random_generator = batch_creators.random(
        batch_size,
        num_samples,
        num_epochs)

    data_object = batch_creators.NumpyData(data)
    if not (labels is None):
        labels_object = batch_creators.NumpyData(labels)

    for indices in random_generator:
        batch = dict()
        if not (labels is None):
            batch['data'] = data_object.get_data(indices)
            batch['labels'] = labels_object.get_data(indices)
            yield batch
                
        else:
            batch['data'] = data_object.get_data(indices)
            yield batch
        


        

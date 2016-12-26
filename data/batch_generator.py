from pyAIUtils.aiutils.data import batch_creators

import numpy as np


def create_batch_generator(data,batch_size,num_epochs,labels=None,seed=None):
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
        if not (labels is None):
            yield data_object.get_data(indices), \
                labels_object.get_data(indices)
        else:
            yield data_object.get_data(indices)
        


        


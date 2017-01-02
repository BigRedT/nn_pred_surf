import decision_functions

import tensorflow as tf
import os


class Constants():
    outdir = '/Users/tanmay/Code/nn_pred_surf_results'
    experiment_name = 'default'
    pred_filename = 'pred'
    gt_filename = 'gt'
    input_dims = 2
    hidden_units = [10]*4
    residual = [False]*4
    activation = 'relu'
    keep_prob = 1.
    use_batchnorm = False
    # is_training = True
    learning_rate = 0.01
    batch_size = 1000
    num_epochs = 100
    num_train_samples = 10000
    domain = [(-2,2),(-2,2)]
    steps = [0.01,0.01]
    decision_funcs = None

    def mkdir_if_not_exists(self,dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    def get_experiments_dir(self):
        experiments_dir =  os.path.join(self.outdir,self.experiment_name)
        self.mkdir_if_not_exists(experiments_dir)
        return experiments_dir    
    

   

import train
import experiments.run_experiment as run_experiment
import decision_functions
import constants

import numpy as np


def main():
    c = constants.Constants()
    c.experiment_name = 'effect_of_num_samples_w_residual'
    c.batch_size = 100
    c.residual = [False, True, True, True]
    num_train_samples = [100,1000,10000,100000]
    

    decision_funcs = dict()
    decision_funcs['line'] = lambda x: decision_functions.ax_b(
        1,
        0,
        x)

    decision_funcs['abs'] = lambda x: decision_functions.absx(x)

    decision_funcs['ellipse'] = lambda x: decision_functions.ellipse(
        2,
        1,
        x)
    
    decision_funcs['sin'] = lambda x: decision_functions.asincx_bcosdx(
        1,
        0,
        np.pi,
        np.pi,
        x)
    
    for i in range(len(num_train_samples)):
        c.num_train_samples = num_train_samples[i]
        for func_type, decision_func in decision_funcs.items():
            c.decision_func = decision_func
            c.pred_filename = 'pred_' + str(c.num_train_samples) + '_' + func_type
            c.gt_filename = 'gt_' + func_type
            run_experiment.run(c)

            
if __name__=='__main__':
    main()

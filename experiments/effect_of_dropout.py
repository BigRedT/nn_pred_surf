import train
import experiments.run_experiment as run_experiment
import decision_functions
import constants

import numpy as np


def main():
    c = constants.Constants()
    c.experiment_name = 'effect_of_dropout_corrected'
    keep_probs = {
        'low' : 0.1,
        'medium' : 0.5,
        'medium_high' : 0.7,
        'high' : 0.9,
    }
    

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
    
    for name, keep_prob in keep_probs.items():
        c.keep_prob = keep_prob
        for func_type, decision_func in decision_funcs.items():
            c.decision_func = decision_func
            c.pred_filename = 'pred_' + name + '_' + func_type
            c.gt_filename = 'gt_' + func_type
            run_experiment.run(c)

            
if __name__=='__main__':
    main()

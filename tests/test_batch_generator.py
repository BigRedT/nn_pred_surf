from data.batch_generator import create_batch_generator
from data.sample import sample_training_data
import decision_functions


def test_create_batch_generator():
    data = sample_training_data(
        10,
        [(-1,1),(-1,1)],
        seed=0)

    labels = decision_functions.ax_b(1,0,data)
    batches = create_batch_generator(data,5,2,labels=labels,seed=1)
    for batch in batches:
        print(batch)
        
    batches = create_batch_generator(data,5,2,seed=1)
    for batch in batches:
        print(batch)

        
def main():
    test_create_batch_generator()

    
if __name__=='__main__':
    main()

from data.batch_generator import create_batch_generator
from data.sample import sample_training_data


def test_create_batch_generator():
    data = sample_training_data(
        10,
        [(-1,1),(-1,1)])
    batches = create_batch_generator(data,5,2)
    for batch in batches:
        print(batch)

        
def main():
    test_create_batch_generator()

    
if __name__=='__main__':
    main()

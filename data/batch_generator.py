from pyAIUtils.aiutils.data import batch_creators


def create_batch_generator(data,batch_size,num_epochs):
    num_samples = data.shape[0]
    random_generator = batch_creators.random(
        batch_size,
        num_samples,
        num_epochs)

    data_object = batch_creators.NumpyData(data)
    for indices in random_generator:
        yield data_object.get_data(indices)



        


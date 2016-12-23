import numpy as np


def sample_training_data(
    num_samples,
    domain):
    # Args:
    #   - num_samples : Number of training samples to generate
    #   - domain : A list of tuples specifying the range along each dimension
    
    # Returns:
    #   - samples : num_samples x len(domain) matrix consisting of samples drawn 
    #       uniformly from the domain
 
    # Example:
    # To draw 10 samples in R^2 where the first dimension ranges between (-1,1)
    # and the second dimension between (2,3), the call looks like

    samples = sample_training_data(10,[(-1,1),(2,3)])

    num_dims = len(domain)
    samples = np.zeros([num_samples,num_dims])
    for i, (l,h) in enumerate(domain):
        samples[:,i] = np.random.uniform(l,h,(num_samples,))
    
    return samples


def sample_grid_data(
    domain,
    steps,
    reshape=True):
    # Args:
    #   - domain : A list of tuples specifying the range along each dimension
    #   - steps : A list specifying step size along each dimension
    #   - reshape : If True, also returns the reshaped grid

    # Returns:
    #   - grid: A list of length len(domain) where grid[i] is an ndarray of 
    #       grid samples along the ith dimension
    #   - samples: Only returned if reshape is True. Returns the grid samples
    #       as a 2D array. 

    # Example:
    # grid, samples = sample_grid_data([(-1,1),(2,3)],[0.1,0.3])

    assert(len(domain)==len(steps)), 'domain and steps need to be of the same size'

    num_dims = len(domain)

    coords = []
    for (l,h), step in zip(domain,steps):
        coords.append(np.arange(l,h,step,np.float32))

    grid = np.meshgrid(*coords,indexing='ij')
    
    if reshape:
        num_samples = 1
        for coord in coords:
            num_samples *= len(coord)
        
        samples = np.zeros([num_samples,num_dims])
        for i in range(len(grid)):
            samples[:,i] = np.reshape(grid[i],-1)
            
        return grid, samples

    return grid


    


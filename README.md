# Easily visualize the effects of neural network architectural choices on 2D data

The code allows you to easily visualize the prediction surface of any neural network for any 2D data. This was written partially for generating results for [one of my blog posts](link here). But in addition I have tried to make it easy to try out various tweaks, either your own or one of the many published on arXiv, for improving training and data efficiency of neural networks.

# Dependencies

- Tensorflow 0.12
- Numpy
- Matplotlib
- [pyAIUtils](https://github.com/BigRedT/pyAIUtils)

# Setup

Install all the above dependencies except pyAIUtils. Clone the repository and run the following commands inside the created directory to setup pyAIUtils which is used as submodule.
```
git submodule init
git submodule update
```
After running these commands, make sure the pyAIUtils directory is created and has aiutils directory inside it.

# Code Overview

See nn_pred_surf/experiments directory for example scripts. Each of these scripts uses the `Constants` class to store network specifications. An object of this class is passed on to `run_experiments.run()` which [samples data](./data), [creates a tensorflow computation graph](./graph.py), trains the network and produces a decision surface.

Visualize effect of:
- Architectural choices
  - Depth
  - Width
  - Simple vs Resnet
  - Sigmoid vs Relu
  - Deterministic vs Dropout vs Batchnorm
- Number of training samples
- Complexity of Decision function
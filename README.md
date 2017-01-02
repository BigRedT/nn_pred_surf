# Easily visualize the effects of neural network architectural choices on 2D data

The code allows you to easily visualize the prediction surface of any neural network for any 2D data. This was written partially for generating results for [one of my blog posts](link here). But in addition I have tried to make it easy to try out various tweaks, either your own or one of the many published on arXiv, for improving training and data efficiency of neural networks.

# Dependencies

- Tensorflow 0.12
- Numpy
- Matplotlib


Visualize effect of:
- Architectural choices
  - Depth
  - Width
  - Simple vs Resnet
  - Sigmoid vs Relu
  - Deterministic vs Dropout vs Batchnorm
- Number of training samples
- Complexity of Decision function
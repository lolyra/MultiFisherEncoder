# MultiFisherEncoder: Open source code for texture image classification using deep neural networks and Fisher Vectors

Python library for Machine Learning (ML) that allows the implementation of Fisher Vector encoding in association with deep Convolutional Neural Networks (CNN).

## Setup

In order to run the code, proceed to install the required packages. 
Before installing, we advise [creating a virtual environment](https://docs.python.org/3/library/venv.html).

If you want to run the code using the examples provided, install the full requirements:

`$ pip install -r example_requirements.txt`

However, if you want to only use the MultiFisherEncoder provided in this package, you can only install essencial packages:

`$ pip install -r encoder_requirements.txt`

## Execution

An example of usage is provided in `usage_example.py`.

In this example we use Pytorch and Timm to perform feature extraction, however, any other method of feature extraction can be passed to the encoder.

Additionally, the example uses Principal Component Analysis (PCA) as dimensionality reduction tool, however, other tools may be used. 
We provide an AutoEncoder and Average Pooling as alternatives in `examples\reducer.py`.

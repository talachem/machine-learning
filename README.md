# machine-learning

This is a machine learning framework built from scratch using [NumPy](https://numpy.org). It’s not meant for production use — this is more of a learning project to better understand the inner workings of AI and machine learning algorithms.

The framework includes:
- A modular neural network system with various layer types (inspired by [PyTorch](https://pytorch.org))
- A flexible, node-based approach to decision trees
- A basic implementation of self-organizing maps (SOMs)

This is a hobbyist/academic project — expect clarity over optimization and curiosity over performance.


## Feature List

The Following list gives an overview of features and their status. Keep in meind, that this was only recently started and that there are many bugs and a lack of optimization.

- Neural Network
  - Linear (implemented, works great)
  - Dropout (implemented, works)
  - Bachnorm (1d works, 2d backward is wonky)
  - Convolution (2D implemented, 1D planned)
    - with 1D convolutions one can analyse time serieses
    - 1D convolutions can be used as simplefied RNN layers
  - Pooling (min/max and avg are implemented)
  - Loss Functions (implemented, should work, but not sure)
  - Optimizers (implemented)
    - sgd, sgd with momentum and nesterov are a bit wonky
    - adagrad, adadelta, rmsprop and adam work properly
  - Activation (implemented, work)
  - Graph (very simple solution implemented)
  - Transposed Convolution (implemented, works)
  - LR Scheduler (implemented)
  - Tensors with autograd (implemented, but too slow to use)
  - Post Training Quantization (implemented)
- Random Forrest
  - DecisionTree (Modular)
    - Impurity Measures (Gini, Entropy, MAE, MSE)
    - Split Algorithms (CART, ID3, C4.5)
    - Feature Preselection
    - Pruning
  - Regressor (implemented, untested)
  - Classifier (implemented)
  - RandomForret
    - Adding different trees
    - Voting Algorithms
  - Boosting (ansatz)
    - AdaBoosting
    - GradientBoosting
  - Pruning (ansatz)
    - reduce error
    - reduce complexity
    - reduce overfitting
- Self-Organizing Map (minimally implemented)
  - Rectangular and Hexagonal Maps
  - finding BMUs (implemented)
  - Neighborhood (very simple)
    - Neighborhood Functions (several implemented)
  - Dataloader from Neural Network (works)
  - learning rate scheduler from Neural Network (works)
  - calculating umatrix (implemented)
  - evaluating the map (minimally implemented)
- all applications can be saved/loaded as/from json files

The network code is up and running, forward and backward propagation works with Linear, Convolution and Activation, with Dropout and Pooling should work. I am pretty sure that there a bugs and mistakes. But on simple test data it already works well.

For random forrests and decision trees I found a nearly complete code. I reworked that code, first of all I renamed all variables, functions etc. to more meaningful names.
Then I refactored to code a lot and made it more Modular, so that users can assamble trees by hand and append them to forrests... makes them less random.

For the Self-Organizing Map I came up with a solution that can work with batches and reuse some code from the neural network. Next up are validation/prediction methods for the SOM.

The code is still rather bare bones in general. There is not to much internal checking within the code, thus the user needs to be rather careful.
For now I am trying to write proper doc-strings and comments. The plan is to have a more coherent naming system and make code more readable to minimize the need for comments.


## Usage

```python
from machineLearning.nn.layer import (
    Linear, Dropout, Flatten, Convolution2D, Unsqueeze,
    Tanh, SoftMax, Sigmoid, SoftPlus, Relu, Elu, LeakyRelu, SoftSign, Identity,
    BatchNorm1D, BatchNorm2D
)
from machineLearning.nn.optim import SGD, SGDMomentum, NesterovMomentum, AdaGrad, AdaDelta, RMSprop, Adam
from machineLearning.nn.scheduler import ExponentialLR, SteppedLR, CyclicalLR
from machineLearning.nn.module import Sequential, Parallel
from machineLearning.nn.loss import CrossEntropyLoss, MSELoss, NLLLoss, MAELoss, FocalLoss
from machineLearning.data import Data
```


## Sources

- Neural Network
  - [ansatz for linear](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)
  - [more on linear](https://towardsdatascience.com/creating-neural-networks-from-scratch-in-python-6f02b5dd911)
  - [generell approach](https://papers-100-lines.medium.com/neural-network-from-scratch-in-100-lines-of-python-code-dd78e20f8796)
  - [ansatz for convolution](https://blog.ca.meron.dev/Vectorized-CNN/)
  - [more on convolution](https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd)
  - [optimizers](https://towardsdatascience.com/neural-network-optimizers-from-scratch-in-python-af76ee087aab)
  - [graph layer](https://github.com/satrialoka/gnn-from-scratch)
  - [more on graphs](https://theaisummer.com/graph-convolutional-networks/)
  - [a complete implementation](https://github.com/Nico-Curti/NumPyNet)
  - [batchnorm](https://github.com/renan-cunha/BatchNormalization)
  - [batchnorm](https://towardsdatascience.com/implementing-batch-normalization-in-python-a044b0369567)
  - [on transposed convolution](https://towardsdatascience.com/what-are-transposed-convolutions-2d43ac1a0771)
  - [focal loss](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075)
  - [regression loss](https://datamonje.com/regression-loss-functions/)
  - [regularization](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)
  - [how to add l1/l2 regularization](https://androidkt.com/how-to-add-l1-l2-regularization-in-pytorch-loss-function/)
  - [activation functions](https://towardsdatascience.com/creating-neural-networks-from-scratch-in-python-6f02b5dd911)
  - [hopfield layer](https://ml-jku.github.io/hopfield-layers/)
  - [more on hopfield](https://github.com/takyamamoto/Hopfield-Network/blob/master/network.py)
  - [hopflied + mlp](https://link.springer.com/chapter/10.1007/3-540-44868-3_22)
  - [rnn - theory](https://www.freecodecamp.org/news/the-ultimate-guide-to-recurrent-neural-networks-in-python/)
  - [rnn - implementation](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85)
  - [rnn - implementation](https://medium.com/@VersuS_/coding-a-recurrent-neural-network-rnn-from-scratch-using-pytorch-a6c9fc8ed4a7)
  - [lstm - implementation](https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091)
  - [autograd](https://www.robots.ox.ac.uk/~tvg/publications/talks/autodiff.pdf)
  - [more on autograd](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)
  - [implementing autograd](https://learnml.today/making-backpropagation-autograd-mnist-classifier-from-scratch-in-Python-5)
  - [quantization](https://arxiv.org/abs/2106.08295)
  - [more quantization](https://intellabs.github.io/distiller/algo_quantization.html)
  - [even more on quantization](https://yyang768osu.github.io/blog/2022/neural-network-quantization/)
- Random Forrest
  - [ansatz for trees](https://insidelearningmachines.com/build-a-decision-tree-in-python/)
  - [more on trees](https://blog.mattbowers.dev/decision-tree-from-scratch)
  - [generel information](https://www.displayr.com/machine-learning-pruning-decision-trees/)
  - [calc feat importance](https://medium.com/data-science-in-your-pocket/how-feature-importance-is-calculated-in-decision-trees-with-example-699dc13fc078)
  - [id3 algorithm](https://towardsdatascience.com/id3-decision-tree-classifier-from-scratch-in-python-b38ef145fd90)
  - [more on id3](https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f)
  - [ansatz for forrest](https://insidelearningmachines.com/build-a-random-forest-in-python/)
  - [about boosting/bagging](https://blog.mlreview.com/gradient-boosting-from-scratch-1e317ae4587d)
  - [ada boosting](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/)
  - [gradient boosting](https://www.machinelearningplus.com/machine-learning/gradient-boosting/)
  - [about pruning](https://12ft.io/proxy?q=https%3A%2F%2Ftowardsdatascience.com%2Fbuild-better-decision-trees-with-pruning-8f467e73b107)
  - [about pruning](https://towardsdatascience.com/build-better-decision-trees-with-pruning-8f467e73b107)
  - [isolation tree](https://towardsdatascience.com/isolation-forest-from-scratch-e7e5978e6f4c)
- Self Organizing Map
  - [basic ansatz](https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/)
  - [more comprehensive guide](https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms)
  - [weight init](https://arxiv.org/pdf/1210.5873.pdf)
  - [step by step guide](https://towardsdatascience.com/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985)
  - [hierarchical maps](https://ieeexplore.ieee.org/document/1058070)
  - [pruning and growing](https://www.hindawi.com/journals/jhe/2022/9972406/)

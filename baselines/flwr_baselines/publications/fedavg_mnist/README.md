# Federated Averaging MNIST

The following baseline replicates the experiments in *Communication-Efficient Learning of Deep Networks from Decentralized Data*, which was the first paper to propose a federated approach to machine learning and demonstrated the FederatedAveraging algorthim on the MNIST dataset.

**Paper Abstract:** 

<center>
<i>Modern mobile devices have access to a wealth
of data suitable for learning models, which in turn
can greatly improve the user experience on the
device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos.
However, this rich data is often privacy sensitive,
large in quantity, or both, which may preclude
logging to the data center and training there using
conventional approaches. We advocate an alternative that leaves the training data distributed on
the mobile devices, and learns a shared model by
aggregating locally-computed updates. We term
this decentralized approach Federated Learning.
We present a practical method for the federated
learning of deep networks based on iterative
model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments
demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a
defining characteristic of this setting. Communication costs are the principal constraint, and
we show a reduction in required communication
rounds by 10–100× as compared to synchronized
stochastic gradient descent</i>
</center>

**Paper Authors:** 

H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.


Note: If you use this implementation in your work, please remember to cite the original authors of the paper. 

**[Link to paper.](https://arxiv.org/pdf/1602.05629.pdf)**

## Running experiments

WIP 

For the moment the experiment can be ran using the `playground.ipynb` notebook.

### Example outputs

WIP

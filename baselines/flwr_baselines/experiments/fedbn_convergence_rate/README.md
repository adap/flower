# FedBN Baseline - Convergence Rate

## Experiment setup



## Dataset 

5 different data sets are used to simulate a non-IID data distribution within 5 clients. The following datasets are used:

* [MNIST](https://ieeexplore.ieee.org/document/726791)
* [MNIST-M]((https://arxiv.org/pdf/1505.07818.pdf))
* [SVHN](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
* [USPS](https://ieeexplore.ieee.org/document/291440)
* [SynthDigits](https://arxiv.org/pdf/1505.07818.pdf)

A more detailed explanation of the datasets are given in the following table. 

|     | MNIST     | MNIST-M   | SVHN  |  USPS    | SynthDigits |
|--- |---        |---        |---    |---            |---    |
| data type| handwritten digits| MNIST modification randomly colored with colored patches| Street view house numbers | handwritten digits from envelopes by the U.S. Postal Service | Syntehtic digits Windows TM font varying the orientation, blur and stroke colors |
| color | greyscale | RGB | RGB | greyscale | RGB |
| pixelsize | 28x28 | 28 x 28 | 32 x32 | 16 x16 | 32 x32 |
| labels | 0-9 | 0-9 | 1-10 | 0-9 | 1-10 |
| number of trainset | 60.000 | 60.000 | 73.257 | 9,298 | 50.000 |
| number of testset| 10.000 | 10.000 | 26.032 | - | 0 |
| image shape | (28,28) | (28,28,3) | (32,32,3) | (16,16) | (32,32,3) |

## Dataset Download

The Research team from the [FedBN paper](https://arxiv.org/pdf/2102.07623.pdf) prepared a pre-processed dataset on their GitHub repository that is available [here](https://github.com/med-air/FedBN). Please download their data, save it in a `/data` directory and unzip afterwards. 
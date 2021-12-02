# FedBN Baseline - Convergence Rate

## Dataset 

|     | MNIST     | MNIST-M   | SVHN  |  USPS    |SynthDigits |
|--- |---        |---        |---    |---            |---    |
| data type| handwritten digits| MNIST modification randomly colored with colored patches| Street view house numbers | handwritten digits from envelopes by the U.S. Postal Service | Syntehtic digits based on SVHN |
| color | greyscale | RGB | RGB | greyscale | RGB |
| pixelsize | 28x28 | 28 x 28 | 32 x32 | 16 x16 | 32 x32 |
| labels | 0-9 | 0-9 | 1-10 | 0-9 | 1-10 |
| number of trainset | 60.000 | 60.000 | 73.257 | 9,298 | 73.257 |
| number of testset| 10.000 | 10.000 | 26.032 | - | 26.032 |
| image shape | (28,28) | (28,28,3) | (32,32,3) | (16,16) | (32,32,3) |
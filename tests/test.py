import numpy as np

from flwr.common.secagg_utils import quantize, reverse_quantize

weights = np.array([[-2, -5, 19], [0, 2, -1]])
weights %= 3
print(weights)

'''# TESTING
vector = secagg_utils.weights_zero_generate(
    [(2, 3), (2, 3)])
for i in ask_vectors_results:
    vector = secagg_utils.weights_addition(vector, parameters_to_weights(
        i[1].parameters))
vector = secagg_utils.weights_mod(vector, mod_range)
vector = secagg_utils.weights_divide(vector, len(ask_vectors_results))
print(vector)
print(secagg_utils.reverse_quantize(vector, clipping_range, target_range))
# TESTING END'''

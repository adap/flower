import timeit
import numpy as np

from flwr.common.sec_agg import sec_agg_test
import random
from flwr.common import weights_to_parameters
from flwr.common.sec_agg.sec_agg_primitives import combine_shares, create_shares
'''weights: Weights = [np.array([[-0.2, -0.5, 1.9], [0.0, 2.4, -1.9]]),
                    np.array([[0.2, 0.5, -1.9], [0.0, -2.4, 1.9]])]
quantized_weights = sec_agg_primitives.quantize(
    weights, 3, 10)
quantized_weights = sec_agg_primitives.weights_divide(quantized_weights, 4)
print(quantized_weights)'''


'''def test_combine_shares() -> None:
    x = timeit.default_timer()
    message = b"Quack quack!"
    share_num = 1000
    threshold = 100
    shares = create_shares(message, threshold, share_num)
    shares_collected = random.sample(shares, threshold)
    message_constructed = combine_shares(shares_collected)
    assert(message == message_constructed)
    y = timeit.default_timer()
    print(y-x)'''


if __name__ == "__main__":
    # test_combine_shares()
    f = open("log.txt", "w")
    f.write("Starting\n")
    f.close()
    sample_num_list = [100, 200, 300, 400, 500]
    dropout_value_list = [0, 1, 2, 3]

    for sample_num in sample_num_list:
        for dropout_value in dropout_value_list:
            for i in range(10):
                f = open("log.txt", "a")
                f.write(
                    f"This is secagg sampling {sample_num} dropping out {dropout_value}0% try {i} \n")
                f.close()
                sec_agg_test.test_start_simulation(
                    sample_num=sample_num, share_num=sample_num, threshold=int(sample_num / 2), dropout_value=dropout_value, num_rounds=1)


'''# TESTING
vector = sec_agg_primitives.weights_zero_generate(
    [(2, 3), (2, 3)])
for i in ask_vectors_results:
    vector = sec_agg_primitives.weights_addition(vector, parameters_to_weights(
        i[1].parameters))
vector = sec_agg_primitives.weights_mod(vector, mod_range)
vector = sec_agg_primitives.weights_divide(vector, len(ask_vectors_results))
print(vector)
print(sec_agg_primitives.reverse_quantize(vector, clipping_range, target_range))
# TESTING END'''

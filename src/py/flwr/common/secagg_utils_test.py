"""Secure Aggregation Utils Tests."""
import base64
import os
import random

from cryptography.fernet import Fernet
import numpy as np
from numpy.core.numerictypes import issubdtype
from numpy.testing._private.utils import assert_almost_equal
from flwr.common.secagg_utils import bytes_to_private_key, bytes_to_public_key, combine_shares, create_shares, decrypt, encrypt, generate_key_pairs, generate_shared_key, private_key_to_bytes, pseudo_rand_gen, public_key_to_bytes, quantize, rand_bytes, reverse_quantize, share_keys_plaintext_concat, share_keys_plaintext_separate, weights_addition, weights_divide, weights_mod, weights_shape, weights_subtraction, weights_zero_generate

# Test if generated shared key is identical


def test_key_generation() -> None:
    sk1, pk1 = generate_key_pairs()
    sk2, pk2 = generate_key_pairs()
    dk1 = generate_shared_key(sk1, pk2)
    dk2 = generate_shared_key(sk2, pk1)
    assert(dk1 == dk2)
    assert(len(base64.urlsafe_b64decode(dk1)) == 32)


def test_serde_private_keys() -> None:
    sk1, pk1 = generate_key_pairs()
    sk2, pk2 = generate_key_pairs()
    sk1_modified = bytes_to_private_key(private_key_to_bytes(sk1))
    dk = generate_shared_key(sk1, pk2)
    dk_modified = generate_shared_key(sk1_modified, pk2)
    assert(dk == dk_modified)


def test_serde_public_keys() -> None:
    sk1, pk1 = generate_key_pairs()
    sk2, pk2 = generate_key_pairs()
    pk1_modified = bytes_to_public_key(public_key_to_bytes(pk1))
    dk = generate_shared_key(sk2, pk1)
    dk_modified = generate_shared_key(sk2, pk1_modified)
    assert(dk == dk_modified)


def test_encrypt_decrypt() -> None:
    key = base64.urlsafe_b64encode(rand_bytes(32))
    message1 = b'Quack quack!'
    message2 = decrypt(key, encrypt(key, message1))
    assert(message1 == message2)


def test_create_shares() -> None:
    message = b"Quack quack!"
    share_num = 10
    threshold = 8
    shares = create_shares(message, threshold, share_num)
    assert(len(shares) == share_num)
    for share in shares:
        assert(isinstance(share, bytes))


def test_combine_shares() -> None:
    message = b"Quack quack!"
    share_num = 10
    threshold = 8
    shares = create_shares(message, threshold, share_num)
    shares_collected = random.sample(shares, threshold)
    message_constructed = combine_shares(shares_collected)
    assert(message == message_constructed)

    try:
        insufficient_shares_collected = random.sample(shares, threshold-1)
        message_constructed = combine_shares(insufficient_shares_collected)
        assert(False)
    except Exception as e:
        pass


def test_rand_bytes():
    r1 = rand_bytes()
    assert(len(r1) == 32)
    r2 = rand_bytes(33)
    assert(len(r2) == 33)


def test_pseudo_rand_gen():
    seed = os.urandom(32)
    num_range = 10
    dimensions_list = [(3, 2), (2, 3, 1)]
    v1 = pseudo_rand_gen(seed, num_range, dimensions_list)
    for idx, elm in enumerate(v1):
        assert(elm.shape == dimensions_list[idx])
        for n in elm.flatten():
            assert(np.issubdtype(type(n), np.integer))
            assert(n < 10 and n >= 0)
    v2 = pseudo_rand_gen(seed, num_range, dimensions_list)
    for idx in range(len(v1)):
        assert(np.array_equal(v1[idx], v2[idx]))


def test_string_concat():
    source = 1
    destination = 10
    msg1 = b'Quack quack'
    msg2 = b'Oink Oink'
    concat_msg = share_keys_plaintext_concat(source, destination, msg1, msg2)
    received_source, received_destination, received_msg1, received_msg2 = share_keys_plaintext_separate(
        concat_msg)
    assert(received_source == source)
    assert(received_destination == destination)
    assert(received_msg1 == msg1)
    assert(received_msg2 == msg2)


def test_quantize():
    weights = [np.array([[-0.2, -0.5, 1.9], [0.0, 2.4, -3.9]]),
               np.array([[0.2, 0.5, -1.9], [0.0, -2.4, 8]])]
    clipping_range = 3
    target_range = 10000
    quantized_weights = quantize(weights, clipping_range, target_range)
    for elm in quantized_weights:
        for n in elm.flatten():
            assert(np.issubdtype(type(n), np.integer))
            assert(n >= 0 and n < target_range)

    received_weights = reverse_quantize(quantized_weights, clipping_range, target_range)
    for idx1, elm in enumerate(received_weights):
        for idx2, n in enumerate(elm.flatten()):
            assert(np.issubdtype(type(n), np.float))
            compare_n = weights[idx1].flatten()[idx2]
            if compare_n < -clipping_range:
                np.testing.assert_almost_equal(-clipping_range, n)
            elif compare_n > clipping_range:
                np.testing.assert_almost_equal(clipping_range, n, 2)
            else:
                np.testing.assert_almost_equal(n, compare_n, 3)


def test_weights_manipulation():
    weights1 = [np.array([[-2, 5, 19], [0, 24, -3]]),
                np.array([[2, 5, -19], [0, -24, 8]])]
    weights2 = [np.array([[-2, 5, 19], [0, 24, -3]]),
                np.array([[2, 5, -19], [0, -24, 8]])]
    assert([(2, 3), (2, 3)] == weights_shape(weights1))
    zero_weights = weights_zero_generate([(2, 3), (2, 3)])
    assert([(2, 3), (2, 3)] == weights_shape(zero_weights))
    for elm in zero_weights:
        for n in elm.flatten():
            assert(n == 0)
    add = weights_addition(weights1, weights2)
    subtract = weights_subtraction(weights1, weights2)
    mod = weights_mod(weights1, 10)
    divide = weights_divide(weights1, 4)
    for idx1, elm in enumerate(weights1):
        for idx2, n in enumerate(elm):
            assert(add[idx1].flatten()[idx2] == weights1[idx1].flatten()
                   [idx2]+weights2[idx1].flatten()[idx2])
            assert(subtract[idx1].flatten()[idx2] == weights1[idx1].flatten()[
                   idx2]-weights2[idx1].flatten()[idx2])
            assert(mod[idx1].flatten()[idx2] == weights1[idx1].flatten()[idx2] % 10)
            assert(divide[idx1].flatten()[idx2] == weights1[idx1].flatten()[idx2]/4)




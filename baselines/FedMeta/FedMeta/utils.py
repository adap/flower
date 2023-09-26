"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import numpy as np
import pickle

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

# def compute_ema(data, smoothingWeight):
#     smoothedData = []
#     last = 0 if len(data) > 0 else float('nan')
#     numAccum = 0
#     debiasWeight = 0
#
#     for d in data:
#         nextVal = d
#         last = last * smoothingWeight + (1 - smoothingWeight) * nextVal
#         numAccum += 1
#         debiasWeight = 1.0 - pow(smoothingWeight, numAccum)
#         smoothedData.append(last / debiasWeight)
#
#     return smoothedData
#
#
# data = [1, 2, 3, 4, 5]
# smoothingWeight = 0.8
# result = compute_ema(data, smoothingWeight)
# print(result)



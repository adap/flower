import galois
import numpy as np
import torch
import copy


def LCC_encoding_with_points(X: np.ndarray, alpha_s, beta_s, GF) -> np.ndarray:
    U = gen_Lagrange_coeffs_galois(alpha_s, beta_s, GF)
    X_LCC = U.dot(X)
    return X_LCC


def LCC_decoding_with_points(f_eval, alpha_s_eval, beta_s, GF):
    U_dec = gen_Lagrange_coeffs_galois(beta_s, alpha_s_eval, GF)
    f_recon = U_dec.dot(f_eval.view(GF))

    return f_recon.view(np.ndarray)


def gen_Lagrange_coeffs_galois(alpha_s, beta_s, GF):
    num_alpha = len(alpha_s)
    num_beta = len(beta_s)
    U = np.zeros((num_alpha, num_beta), dtype=np.int64).view(GF)
    alpha_s = GF(alpha_s)
    beta_s = GF(beta_s)
    mbeta = -beta_s

    den = np.zeros(num_beta, dtype="int64").view(GF)
    msk = np.ones(num_beta, dtype=np.bool)

    for j in range(num_beta):
        msk[j] = False
        msk[j - 1] = True
        den[j] = (mbeta + beta_s[j])[msk].prod()

    l = np.zeros(num_alpha, dtype="int64").view(GF)
    for i in range(num_alpha):
        l[i] = (mbeta + alpha_s[i]).prod()
    for i in range(num_alpha):
        for j in range(num_beta):
            U[i][j] = l[i] / ((alpha_s[i] + mbeta[j]) * den[j])
    return U


def model_masking(weights, local_mask: np.ndarray, GF):
    pos = 0
    local_mask = local_mask.view(GF)
    weights = [w.view(GF) for w in weights]
    for i, w in enumerate(weights):
        d = w.size
        cur_mask = local_mask[pos: pos + d]
        cur_mask = cur_mask.reshape(w.shape)
        w += cur_mask
        pos += d
    return weights


def model_unmasking(weights, aggregated_mask: np.ndarray, GF):
    pos = 0
    msk = aggregated_mask.view(GF)
    for i in range(len(weights)):
        w = weights[i]
        d = w.size
        cur_mask = msk[pos: pos + d]
        cur_mask = cur_mask.reshape(w.shape)
        w -= cur_mask
        weights[i] = weights[i].view(np.ndarray)
        pos += d
    return weights


def mask_encoding(
    total_dimension, num_clients, targeted_number_active_clients, privacy_guarantee, galois_field, local_mask: np.ndarray
) -> np.ndarray:
    d = total_dimension
    N = num_clients
    U = targeted_number_active_clients
    T = privacy_guarantee
    p = galois_field.order

    alpha_s = list(range(1, N + 1))
    beta_s = list(range(N + 1, N + U + 1))

    n_i = np.random.randint(p, size=(T * (d // (U - T)), 1))
    LCC_in = np.concatenate([local_mask, n_i], axis=0)
    LCC_in = np.reshape(LCC_in, (U, d // (U - T)))
    encoded_mask_set = LCC_encoding_with_points(LCC_in.view(galois_field), alpha_s, beta_s, galois_field)

    return encoded_mask_set


def compute_aggregate_encoded_mask(encoded_mask_dict, GF, active_clients):
    aggregate_encoded_mask = np.zeros_like(encoded_mask_dict[active_clients[0]]).view(GF)
    for client_id in active_clients:
        aggregate_encoded_mask += encoded_mask_dict[client_id].view(GF)
    return aggregate_encoded_mask


def aggregate_models_in_finite(weights_finite, prime_number):
    '''
    weights_finite : array of state_dict()
    prime_number   : size of the finite field
    '''
    w_sum = copy.deepcopy(weights_finite[0])

    for key in w_sum.keys():

        for i in range(1, len(weights_finite)):
            w_sum[key] += weights_finite[i][key]
            w_sum[key] = np.mod(w_sum[key], prime_number)

    return w_sum

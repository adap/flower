"""Utility functions."""

from dataclasses import dataclass
from typing import List

import numpy as np
from flwr.common import NDArrays
from sklearn.mixture import GaussianMixture


@dataclass
class GMMParameters:
    """GMM parameters."""

    label: int
    means: NDArrays
    weights: NDArrays
    covariances: NDArrays
    num_samples: int


def gmmparam_to_ndarrays(gmm: GMMParameters) -> NDArrays:
    """Convert gmm object to NumPy ndarrays."""
    return [gmm.label, gmm.means, gmm.weights, gmm.covariances, gmm.num_samples]


def ndarrays_to_gmmparam(ndarrays: NDArrays) -> GMMParameters:
    """Convert NumPy ndarray to GMM object."""
    return GMMParameters(
        label=ndarrays[0],
        means=ndarrays[1],
        weights=ndarrays[2],
        covariances=ndarrays[3],
        num_samples=ndarrays[4],
    )


def learn_gmm(
    features: np.array,
    labels: np.array,
    n_mixtures: int,
    cov_type: str,
    seed: int,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> List[GMMParameters]:
    """Learn a list of 16-bits GMMs for each label.

    Parameters
    ----------
    features : np.array
        A 2-d array with size (n_samples, feature_dimension) containing
        extracted features for all the samples.
    labels : np.array
        An array with size (n_samples) containing labels associated for
        each sample in `features`.
    n_mixtures : int
        Number of mixtures in each Gaussian Mixture.
    cov_type : str
        Covariance type of Gaussian Mixtures, e.g. spherical.
    seed: int
        Seed for learning and sampling from Gaussian Mixtures.
    tol: float
        Tolerance of Gaussian Mixtures.
    max_iter: int
        Number of maximum iterations to learn the Gaussian Mixtures.

    Returns
    -------
    List[GMMParameters]
        Returns a list containing the GMMParameters for each class.
    """
    gmm_list = []
    for label in np.unique(labels):
        cond_features = features[label == labels]
        if (
            len(cond_features) > n_mixtures
        ):  # number of samples should be larger than `n_mixtures`.
            gmm = GaussianMixture(
                n_components=n_mixtures,
                covariance_type=cov_type,
                random_state=seed,
                tol=tol,
                max_iter=max_iter,
            )
            gmm.fit(cond_features)
            gmm_list.append(
                GMMParameters(
                    label=label,
                    means=gmm.means_.astype("float16"),
                    weights=gmm.weights_.astype("float16"),
                    covariances=gmm.covariances_.astype("float16"),
                    num_samples=len(cond_features),
                )
            )
    return gmm_list


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

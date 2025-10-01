"""Differential Privacy Utilities for TraceFL."""

import logging

import numpy as np

from flwr.common import NDArrays

# Get logger (logging is configured centrally in server_app.py)
logger = logging.getLogger(__name__)


def safe_clip_inputs_inplace(model_update, clipping_norm):
    """Safely clip model updates in-place with handling for zero norm cases.

    Args:
        model_update: List of model parameter updates
        clipping_norm: Maximum L2 norm for clipping
    """
    try:
        # Validate inputs
        if not model_update or not isinstance(model_update, list):
            logger.error("Invalid model update format")
            return

        if clipping_norm <= 0:
            logger.error("Invalid clipping norm: %f", clipping_norm)
            return

        # Compute L2 norm of the update
        input_norm = compute_l2_norm(model_update)
        logger.info("Input norm before clipping: %.6f", input_norm)

        # Handle zero norm case
        if input_norm == 0:
            logger.warning("Model update has zero norm, skipping clipping")
            return

        # Clip if norm exceeds threshold
        if input_norm > clipping_norm:
            scaling_factor = clipping_norm / input_norm
            logger.info("Applying clipping with scaling factor: %.6f", scaling_factor)
            for i, update in enumerate(model_update):
                model_update[i] = update * scaling_factor

            # Verify clipping
            final_norm = compute_l2_norm(model_update)
            logger.info("Final norm after clipping: %.6f", final_norm)

    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.error("Error in safe_clip_inputs_inplace: %s", str(e))
        raise


def compute_l2_norm(inputs):
    """Compute L2 norm of input arrays.

    Args:
        inputs: List of numpy arrays

    Returns
    -------
        float: L2 norm of the concatenated arrays
    """
    try:
        if not inputs:
            logger.warning("Empty input list provided to compute_l2_norm")
            return 0.0

        # Validate input types
        if not all(isinstance(x, np.ndarray) for x in inputs):
            logger.error("Invalid input type - all inputs must be numpy arrays")
            return 0.0

        # Compute norm using the same approach as Flower's get_norm
        array_norms = [np.linalg.norm(array.flat) for array in inputs]
        total_norm = float(np.sqrt(sum(norm**2 for norm in array_norms)))
        return total_norm

    except (ValueError, TypeError) as e:
        logger.error("Error computing L2 norm: %s", str(e))
        return 0.0


def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    try:
        if not input_arrays:
            logger.warning("Empty input arrays provided to get_norm")
            return 0.0

        array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
        total_norm = float(np.sqrt(sum(norm**2 for norm in array_norms)))
        return total_norm

    except (ValueError, TypeError) as e:
        logger.error("Error in get_norm: %s", str(e))
        return 0.0

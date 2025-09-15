import numpy as np
from flwr.proto import fedmd_pb2

DTYPE_TO_STR = {
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("float16"): "float16",
    np.dtype("int64"): "int64",
    np.dtype("int32"): "int32",
}

STR_TO_DTYPE = {v: k for k, v in DTYPE_TO_STR.items()}

def ndarray_to_tensor(arr: np.ndarray) -> fedmd_pb2.Tensor:
    arr_c = np.ascontiguousarray(arr)
    return fedmd_pb2.Tensor(
        shape=list(arr_c.shape),
        buffer=arr_c.tobytes(),
        dtype=DTYPE_TO_STR.get(arr_c.dtype, str(arr_c.dtype)),
    )

def tensor_to_ndarray(t: fedmd_pb2.Tensor) -> np.ndarray:
    dtype = STR_TO_DTYPE.get(t.dtype, np.dtype(t.dtype))
    arr = np.frombuffer(t.buffer, dtype=dtype).reshape(t.shape)
    return arr

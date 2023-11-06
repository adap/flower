import json
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


# Serialization
numPyData = {
    "id": 25,
    "floatSample": np.float32(1.2),
    "intSample": np.int32(42),
    "arangeSample": np.arange(12),
}
encodedNumpyData = json.dumps(numPyData, cls=NumpyArrayEncoder)
decodedArrays = json.loads(encodedNumpyData)
if __name__ == "__main__":
    print(type(encodedNumpyData))
    print(type(numPyData))
    print((encodedNumpyData))
    print((numPyData))

"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from sklearn.metrics import mean_squared_error, accuracy_score

def evaluate(task_type,y,preds):
    if task_type.upper() == "BINARY":
        result = accuracy_score(y, preds)
    elif task_type.upper() == "REG":
        result = mean_squared_error(y, preds)
    return result
        
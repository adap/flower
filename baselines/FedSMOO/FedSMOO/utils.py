"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_fn(cfg, acc: List[Tuple[int, float]]):
    x_val = [x[0] for x in acc]
    y_val = [x[1] for x in acc]

    plt.plot(x_val,y_val,'or')

    if cfg.dataset.dirichlet:
        dataset_prep = "dirichlet"
    else:
        dataset_prep = "pathological"
    fname = f"./results/{cfg.method}_{cfg.dataset.dataset_name}_{cfg.num_clients}_{dataset_prep}.png"
    plt.savefig(fname)

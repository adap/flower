import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    hist = np.load("_static/results/hist.npy", allow_pickle=True).item()
    rounds, values = zip(*hist.metrics_distributed_fit["accuracy"])
    plt.plot(np.asarray(rounds), np.asarray(values))
    plt.savefig("_static/results/accuracy.png")

import matplotlib.pyplot as plt
import pickle 
import os
from flwr.server.history import History

import pandas as pd

def plot_results(results, title, xlabel, ylabel, legend, save_path):
    plt.figure()
    for result in results:
        plt.plot(result)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.savefig(save_path)
    plt.close()

def open_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # use directories if they start with mobile
    directories = [d for d in os.listdir('./use') if d.startswith('mobile')]
    
    results_for_plot = []
    for directory in directories:
        result_path = os.path.join('./use', directory, 'results.pkl')
        print(directory)
        results = open_pickle(result_path)
        history = results['history']
        
        metric_type = "distributed"
        metric_dict = (history.metrics_distributed)
        _, values = zip(*metric_dict["accuracy"])

        results_for_plot.append(values)
        
    keys = [
        'FedPer #class-10', 'FedAvg #class-8', 'FedAvg #class-10',
        'FedAvg #class-4', 'FedPer #class-8', 'FedPer #class-4',
        ]
    
    plt.figure()
    for i, result in enumerate(results_for_plot):
        # set y axis to cross 0
        plt.xlim(left=0)
        plt.xlim(right=50)
        label = keys[i]
        if 'fedavg' in label.lower():
            plt.plot(result, label=keys[i], linestyle='dashed')
        else:   
            plt.plot(result, label=keys[i])
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig('./use/mobile_plot_figure_2.png')
    plt.show()

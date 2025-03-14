
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re

def clean_results(results_pkl: str, metric: str):
    with open(results_pkl, 'rb') as file:
        results = pickle.load(file)

    if metric == "accuracy":
        first = getattr(results, "metrics_centralized")  
        getval = first["accuracy"]    
    else:
        getval = getattr(results, metric)  
    val_vec = [item for sublist in getval for item in sublist]
    val = np.array(val_vec)
    val.resize(int(val.shape[0]/2),2)

    return val

def plot_results(input, metric, labels: str, color: str = ['g','b','r','k','y','m']):
    for i, v in enumerate(input):
        results = clean_results(v, metric)
        plt.plot(results[0:,0],results[0:,1],marker='', linestyle='-', color=color[i], label=labels[i])

    plt.xlabel('communication round')
    plt.ylabel(metric)
    if metric == "accuracy":
        plt.ylim(0.0,1.0)
    else:
        plt.ylim(0.4,1.5)
    plt.legend()
    plt.grid(True)
    plt.show()

def clean_crossval_results(results_pkl: str, metric: str):
    with open(results_pkl, 'rb') as file:
        results = pickle.load(file)

    if metric == "accuracy":
        first = getattr(results, "metrics_centralized")  
        getval = first["accuracy"]    
    else:
        getval = getattr(results, metric)  
    val_vec = [item for sublist in getval for item in sublist]
    val = np.array(val_vec)
    val.resize(int(val.shape[0]/2),2)

    return val, results_pkl

def calculate_stats(group_results):
    stats = {}
    for group, results in group_results.items():
        # Convert the list of results to a numpy array for easy calculation
        results_array = np.array(results)
        
        # Calculate mean and std deviation along each column (assuming each result is a list of values)
        mean_values = np.mean(results_array, axis=0)
        std_values = np.std(results_array, axis=0)
        
        stats[group] = {'mean': mean_values, 'std': std_values}
    
    print(stats)
    return stats


def plot_crossval_results(input, metric, labels: str, color: str = ['g','b','r','k','y','m','c']):
    numkeep_pattern = r'numkeep(\d+)'
    group_results = {}
    for i, v in enumerate(input):
        results, name = clean_crossval_results(v, metric)
        # group results by agg method and numkeep
        if 'fedavg' in name:
            if 'fedavg' not in group_results:
                group_results['fedavg'] = [] 

            group_results['fedavg'].append(results)
        else:
            match = re.search(numkeep_pattern, name)
            if match:
                numkeep_value = match.group(1)
            group_key = f'fedht_numkeep_{numkeep_value}'
            if group_key not in group_results:
                group_results[group_key] = []
            group_results[group_key].append(results)
        
    print(group_results.keys())

    # Calculate stats (mean and stdev) for each group
    stats = calculate_stats(group_results)

    for i, item in enumerate(stats.items()):
        group, results = item
        # print(group)
        mean = stats[group]["mean"][:,1]
        stdev = stats[group]["std"][:,1]
        # print(stdev)
        plt.plot(stats[group]["mean"][:,0], mean,marker='', linestyle='-', color=color[i], label=labels[i])
        plt.fill_between(stats[group]["mean"][:,0], mean - 3*stdev, mean + 3*stdev, color=color[i], alpha=0.2)

    plt.xlabel('communication round')
    plt.ylabel(metric)
    if metric == "accuracy":
        plt.ylim(0.0,1.0)
    else:
        plt.ylim(0.4,1.5)
    plt.legend()
    plt.grid(True)
    plt.show()

input1 = ['mnist_fedavg_local1_lr1e-05_numkeep500.pkl',
          'mnist_fedht_local1_lr1e-05_numkeep500.pkl',
          'mnist_fedhtiter_local1_lr1e-05_numkeep500.pkl',
         ]

input2 = ['mnist_fedavg_local5_lr1e-05_numkeep500.pkl',
          'mnist_fedht_local5_lr1e-05_numkeep500.pkl',
          'mnist_fedhtiter_local5_lr1e-05_numkeep500.pkl',
         ]

plot_results(input1, 'losses_centralized', ['FedAvg (1 local epoch)','Fed-HT (1 local epoch)', 'FedIter-HT (1 local epoch)'])
plot_results(input1, 'accuracy', ['FedAvg (1 local epoch)','Fed-HT (1 local epoch)', 'FedIter-HT (1 local epoch)'])

plot_results(input2, 'losses_centralized', ['FedAvg (5 local epochs)', 'Fed-HT (5 local epochs)', 'FedIter-HT (5 local epochs)'])
plot_results(input2, 'accuracy', ['FedAvg (5 local epochs)', 'Fed-HT (5 local epochs)', 'FedIter-HT (5 local epochs)'])

crossval_input = ['mnist_fedavg_local1_lr1e-05_wd0.0_numkeep500_fold1.pkl',
                  'mnist_fedavg_local1_lr1e-05_wd0.0_numkeep500_fold2.pkl',
                  'mnist_fedavg_local1_lr1e-05_wd0.0_numkeep500_fold3.pkl',
                  'mnist_fedavg_local1_lr1e-05_wd0.0_numkeep500_fold4.pkl',
                  'mnist_fedavg_local1_lr1e-05_wd0.0_numkeep500_fold5.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep700_fold1.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep700_fold2.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep700_fold3.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep700_fold4.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep700_fold5.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep500_fold1.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep500_fold2.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep500_fold3.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep500_fold4.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep500_fold5.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep250_fold1.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep250_fold2.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep250_fold3.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep250_fold4.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep250_fold5.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep100_fold1.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep100_fold2.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep100_fold3.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep100_fold4.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep100_fold5.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep50_fold1.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep50_fold2.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep50_fold3.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep50_fold4.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep50_fold5.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep25_fold1.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep25_fold2.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep25_fold3.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep25_fold4.pkl',
                  'mnist_fedht_local1_lr1e-05_wd0.0_numkeep25_fold5.pkl']

plot_crossval_results(crossval_input, 'losses_centralized', ['FedAvg',
                                                             'Fed-HT (numkeep = 700)',
                                                             'Fed-HT (numkeep = 500)',
                                                             'Fed-HT (numkeep = 250)',
                                                             'Fed-HT (numkeep = 100)',
                                                             'Fed-HT (numkeep = 50)',
                                                             'Fed-HT (numkeep = 25)'])
plot_crossval_results(crossval_input, 'accuracy', ['FedAvg',
                                                             'Fed-HT (numkeep = 700)',
                                                             'Fed-HT (numkeep = 500)',
                                                             'Fed-HT (numkeep = 250)',
                                                             'Fed-HT (numkeep = 100)',
                                                             'Fed-HT (numkeep = 50)',
                                                             'Fed-HT (numkeep = 25)'])


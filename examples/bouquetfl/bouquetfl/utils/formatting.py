import pickle
import numpy as np

def format_results():
    try:
        with open("checkpoints/load_and_training_times.pkl", "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        return None
    results["total_load_time"] = np.nan
    results["total_train_time"] = np.nan

    num_rounds = int(results.shape[1] / 2 - 2)

    time_errors = []
    for client_id in range(results.shape[0]):
        values = []
        for round in range(num_rounds):
            values.append(results[f"train_time_{round+1}"][client_id])

        time_errors.append(np.std(values))

    for _ in range(results.shape[0]):
        train_col_list = [f"train_time_{j}" for j in range(1, num_rounds+1)]
        load_col_list = [f"load_time_{j}" for j in range(1, num_rounds+1)]
        results["total_load_time"] = results[load_col_list].sum(axis=1)
        results["total_train_time"] = results[train_col_list].sum(axis=1)
    
    for i in range(num_rounds):
        del results[f"train_time_{i+1}"]
        del results[f"load_time_{i+1}"]

    return results


def format_bar(value, max_value, width=25, fill_char="█"):
    filled = int((value / max_value) * width)
    return fill_char * filled + " " * (width - filled)


def print_timings():
    results = format_results()
    if results is None:
        print("No timing results found.")
        return None
    max_load = max(results["total_load_time"])
    max_train = max(results["total_train_time"])
    max_cpu_name_length = max(results["cpu"].str.len().max() + 2, 0)
    max_gpu_name_length = max(results["gpu"].str.len().max() + 2, 0)
    for i_, entry in results.iterrows():
        load_bar = format_bar(entry["total_load_time"], max_load)
        train_bar = format_bar(entry["total_train_time"], max_train)
        print(
            f"{"\033[32m"}REPORT{"\033[0m"} : Client {i_:<2} |   "
            f"{entry['cpu']:<{max_cpu_name_length}}"
            f"{load_bar} {entry['total_load_time']:6.2f}s ({entry['total_load_time']/max_load*100:5.1f}%) "
            f" |   {entry['gpu']:<{max_gpu_name_length}}"
            f"{train_bar} {entry['total_train_time']:6.2f}s ({entry['total_train_time']/max_train*100:5.1f}%) "
        )

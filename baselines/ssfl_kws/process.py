import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

result_path = './output/result'
save_format = 'pdf'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(mode):
    data = ['SpeechCommandsV1', 'SpeechCommandsV2']
    model = ['tcresnet18']
    if mode == 'fs':
        control_name = [[data, model, ['fs'], ['basic']]]
        controls = make_controls(control_name)
    elif mode == 'ps':
        control_name = [[data, model, ['250', '2500'], ['basic']]]
        controls = make_controls(control_name)
    elif mode == 'semi':
        control_name = [[data, model, ['250', '2500'], ['basic=basic-spec'], ['fix']]]
        controls = make_controls(control_name)
    elif mode == 'semi-aug':
        control_name = [[data, model, ['250'], ['basic=basic', 'basic=basic-rand', 'basic=basic-rands',
                                                'basic=basic-spec-rands'], ['fix']]]
        controls = make_controls(control_name)
    elif mode == 'semi-loss':
        control_name = [[data, model, ['250', '2500'], ['basic=basic-spec'], ['fix-mix']]]
        controls = make_controls(control_name)
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    modes = ['fs', 'ps', 'semi', 'semi-aug', 'semi-loss']
    controls = []
    for mode in modes:
        controls += make_control_list(mode)
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_exp(extracted_processed_result_exp)
    df_history = make_df_history(extracted_processed_result_history)
    make_vis(df_exp, df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    metric_name_1 = ['Loss', 'Accuracy']
    metric_name_2 = ['PAccuracy', 'MAccuracy', 'LabelRatio']
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for k in base_result['logger']['train'].mean:
                mode, metric_name_i = k.split('/')
                if (mode == 'test' and metric_name_i in metric_name_1) or (
                        mode == 'train' and metric_name_i in metric_name_2):
                    if metric_name_i not in processed_result_history:
                        processed_result_history[metric_name_i] = {'history': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name_i]['history'][exp_idx] = \
                        base_result['logger']['train'].history[k]
            for k in base_result['logger']['train'].mean:
                mode, metric_name_i = k.split('/')
                if mode == 'test' and metric_name_i in metric_name_1:
                    if metric_name_i not in processed_result_exp:
                        processed_result_exp[metric_name_i] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_exp[metric_name_i]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = [x for x in processed_result[pivot] if x is not None]
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = [x for x in processed_result[pivot] if x is not None and len(x) > 100]
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def write_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        df_name = '_'.join(control)
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=[0]))
    write_xlsx('{}/result_exp.xlsx'.format(result_path), df)
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        for k in extracted_processed_result_history[exp_name]:
            df_name = '_'.join([*control, k])
            df[df_name].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=[0]))
    write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    return df


def make_vis(df_exp, df_history):
    label_loc_dict = {'Accuracy': 'lower right', 'LabelRatio': 'lower right'}
    fontsize = {'legend': 10, 'label': 12, 'ticks': 12}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    ax_dict_2 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        metric_name, stat = df_name_list[-2], df_name_list[-1]
        semi_mask = len(df_name_list) == 7 and metric_name not in ['Loss'] and stat == 'mean'
        if semi_mask:
            label_dict = {'Accuracy': 'Test', 'PAccuracy': 'Pseudo-Label', 'MAccuracy': 'Thresholded',
                          'LabelRatio': 'Label Ratio',
                          'fs': 'Fully Supervised', 'ps': 'Partially Supervised'}
            color_dict = {'Accuracy': 'red', 'PAccuracy': 'dodgerblue', 'MAccuracy': 'blue', 'LabelRatio': 'green',
                          'fs': 'black', 'ps': 'orange'}
            linestyle_dict = {'Accuracy': '-', 'PAccuracy': '--', 'MAccuracy': ':', 'LabelRatio': '-.',
                              'fs': (0, (5, 5)),
                              'ps': (0, (5, 10))}
            xlabel = 'Epoch'
            ylabel_1 = 'Accuracy'
            ylabel_2 = 'Label Ratio'
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            fig_name = '_'.join([*df_name_list[:-2]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
                ax_dict_2[fig_name] = ax_dict_1[fig_name].twinx()
            ax_1 = ax_dict_1[fig_name]
            ax_2 = ax_dict_2[fig_name]
            y = df_history[df_name].iloc[0].to_numpy()
            y_err = df_history[df_name_std].iloc[0].to_numpy()
            if metric_name in ['PAccuracy', 'MAccuracy', 'LabelRatio']:
                y = y[::2]
                y_err = y_err[::2]
            x = np.arange(len(y))
            if metric_name in ['LabelRatio']:
                ax_2.plot(x, y, label=label_dict[metric_name], color=color_dict[metric_name],
                          linestyle=linestyle_dict[metric_name])
                ax_2.fill_between(x, (y - y_err), (y + y_err), color=color_dict[metric_name], alpha=.1)
            else:
                ax_1.plot(x, y, label=label_dict[metric_name], color=color_dict[metric_name],
                          linestyle=linestyle_dict[metric_name])
                ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[metric_name], alpha=.1)
            if metric_name in ['Accuracy']:
                fs_df_name = '_'.join([*df_name_list[:2], 'fs', 'basic'])
                fig[fig_name] = plt.figure(fig_name)
                x = np.arange(len(y))
                y = df_exp[fs_df_name]['Accuracy_mean'].to_numpy()
                y_err = df_exp[fs_df_name]['Accuracy_std'].to_numpy()
                y = np.repeat(y, len(x))
                y_err = np.repeat(y_err, len(x))
                ax_1.plot(x, y, label=label_dict['fs'], color=color_dict['fs'], linestyle=linestyle_dict['fs'])
                ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict['fs'], alpha=.1)
                ps_df_name = '_'.join([*df_name_list[:3], 'basic'])
                y = df_exp[ps_df_name]['Accuracy_mean'].to_numpy()
                y_err = df_exp[ps_df_name]['Accuracy_std'].to_numpy()
                y = np.repeat(y, len(x))
                y_err = np.repeat(y_err, len(x))
                ax_1.plot(x, y, label=label_dict['ps'], color=color_dict['ps'],
                          linestyle=linestyle_dict['ps'])
                ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict['ps'], alpha=.1)
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel(ylabel_1, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.set_ylabel(ylabel_2, fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        control = fig_name.split('_')
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        handles_1, labels_1 = ax_dict_1[fig_name].get_legend_handles_labels()
        if len(control) == 5:
            handles_2, labels_2 = ax_dict_2[fig_name].get_legend_handles_labels()
            handles = handles_1 + handles_2
            labels = labels_1 + labels_2
            dir_name = 'semi'
        ax_dict_1[fig_name].legend(handles, labels, loc=label_loc_dict['Accuracy'], fontsize=fontsize['legend'])
        fig[fig_name].tight_layout()
        dir_path = os.path.join(vis_path, dir_name)
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()

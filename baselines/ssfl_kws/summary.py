import argparse
from collections import OrderedDict
import models
import os
from config import update_cfg, cfg
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import fetch_dataset, make_data_loader
from utils import save, makedir_exist_ok, to_device, process_control, process_dataset, collate
import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
cudnn.benchmark = True

def process_control_name():
    if 'control_name' in cfg and cfg['control_name'] is not None:
        control_name_list = cfg['control_name'].split('_')
        control_keys_list = list(cfg['control'].keys())
        cfg['control'] = {control_keys_list[i]: control_name_list[i] for i in range(len(control_name_list))}
    if cfg['control'] is not None:
        cfg['control_name'] = '_'.join([str(cfg['control'][k]) for k in cfg['control']])
    return


@hydra.main(version_base=None,config_path=".", config_name="config")
def main(c : DictConfig) -> None:
    update_cfg(dict(c))
    process_control_name()
    pprint(cfg)
    process_control()
    assert cfg == globals()['cfg']

    cfg['seed'] = 0
    runExperiment()
    return


def runExperiment():
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    cfg['summary'] = {}
    cfg['summary']['batch_size'] = {'train': 2, 'test': 2}
    cfg['summary']['shuffle'] = {'train': False, 'test': False}
    data_loader = make_data_loader(dataset, 'summary')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    summary = summarize(data_loader['train'], model)
    content, total = parse_summary(summary)
    print(content)
    save_result = total
    save(save_result, './output/result/{}.pt'.format(cfg['control_name']))
    return


def make_size(input, output):
    if isinstance(input, (tuple, list)):
        return make_size(input[0], output)
    if isinstance(output, (tuple, list)):
        return make_size(input, output[0])
    input_size, output_size = list(input.size()), list(output.size())
    return input_size, output_size


def make_flops(module, input, output):
    if isinstance(input, tuple):
        return make_flops(module, input[0], output)
    if isinstance(output, tuple):
        return make_flops(module, input, output[0])
    flops = compute_flops(module, input, output)
    return flops


def summarize(data_loader, model):
    def register_hook(module):

        def hook(module, input, output):
            module_name = str(module.__class__.__name__)
            if module_name not in summary['count']:
                summary['count'][module_name] = 1
            else:
                summary['count'][module_name] += 1
            key = str(hash(module))
            if key not in summary['module']:
                summary['module'][key] = OrderedDict()
                summary['module'][key]['module_name'] = '{}_{}'.format(module_name, summary['count'][module_name])
                summary['module'][key]['input_size'] = []
                summary['module'][key]['output_size'] = []
                summary['module'][key]['params'] = {}
                summary['module'][key]['flops'] = make_flops(module, input, output)
            input_size, output_size = make_size(input, output)
            summary['module'][key]['input_size'].append(input_size)
            summary['module'][key]['output_size'].append(output_size)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if name in ['weight']:
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params']['weight'] = {}
                            summary['module'][key]['params']['weight']['size'] = list(param.size())
                            summary['module'][key]['coordinates'] = []
                            summary['module'][key]['params']['weight']['mask'] = torch.zeros(
                                summary['module'][key]['params']['weight']['size'], dtype=torch.long)
                    elif name in ['bias']:
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params']['bias'] = {}
                            summary['module'][key]['params']['bias']['size'] = list(param.size())
                            summary['module'][key]['params']['bias']['mask'] = torch.zeros(
                                summary['module'][key]['params']['bias']['size'], dtype=torch.long)
                    else:
                        continue
            if len(summary['module'][key]['params']) == 0:
                return
            for name in summary['module'][key]['params']:
                summary['module'][key]['params'][name]['mask'] += 1
            return

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) \
                and not isinstance(module, nn.ModuleDict) and module != model:
            hooks.append(module.register_forward_hook(hook))
        return

    run_mode = True
    summary = OrderedDict()
    summary['module'] = OrderedDict()
    summary['count'] = OrderedDict()
    hooks = []
    model.train(run_mode)
    model.apply(register_hook)
    for i, input in enumerate(data_loader):
        input = collate(input)
        input = to_device(input, cfg['device'])
        model(input)
        break
    for h in hooks:
        h.remove()
    summary['total_num_params'] = 0
    summary['total_num_flops'] = 0
    for key in summary['module']:
        num_params = 0
        num_flops = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
            num_flops += summary['module'][key]['flops']
        summary['total_num_params'] += num_params
        summary['total_num_flops'] += num_flops
    summary['total_space'] = summary['total_num_params'] * 32. / 8 / (1024 ** 2.)
    return summary


def divide_by_unit(value):
    if value > 1e9:
        return '{:.6} G'.format(value / 1e9)
    elif value > 1e6:
        return '{:.6} M'.format(value / 1e6)
    elif value > 1e3:
        return '{:.6} K'.format(value / 1e3)
    return '{:.6}'.format(value / 1.0)


def parse_summary(summary):
    content = ''
    headers = ['Module Name', 'Input Size', 'Weight Size', 'Output Size', 'Parameters', 'FLOPs']
    records = []
    for key in summary['module']:
        if not summary['module'][key]['params']:
            continue
        module_name = summary['module'][key]['module_name']
        input_size = str(summary['module'][key]['input_size'])
        weight_size = str(summary['module'][key]['params']['weight']['size']) if (
                'weight' in summary['module'][key]['params']) else 'N/A'
        output_size = str(summary['module'][key]['output_size'])
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
        num_flops = divide_by_unit(summary['module'][key]['flops'])
        records.append([module_name, input_size, weight_size, output_size, num_params, num_flops])
    total_num_param = '{} ({})'.format(summary['total_num_params'], divide_by_unit(summary['total_num_params']))
    total_num_flops = '{} ({})'.format(summary['total_num_flops'], divide_by_unit(summary['total_num_flops']))
    total_space = summary['total_space']
    total = {'num_params': summary['total_num_params'], 'num_flops': summary['total_num_flops'],
             'space': summary['total_space']}
    table = tabulate(records, headers=headers, tablefmt='github')
    content += table + '\n'
    content += '================================================================\n'
    content += 'Total Number of Parameters: {}\n'.format(total_num_param)
    content += 'Total Number of FLOPs: {}\n'.format(total_num_flops)
    content += 'Total Space (MB): {:.2f}\n'.format(total_space)
    makedir_exist_ok('./output')
    content_file = open('./output/summary.md', 'w')
    content_file.write(content)
    content_file.close()
    return content, total


def compute_flops(module, inp, out):
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
        return compute_Norm_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.GELU)):
        return compute_ReLU_flops(module, inp, out)
    elif isinstance(module, nn.Upsample):
        return compute_Upsample_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    else:
        print(f"[Flops]: {type(module).__name__} is not supported!")
        return 0


def compute_Conv2d_flops(module, inp, out):
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups
    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w
    total_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count
    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_Norm_flops(module, inp, out):
    assert isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm))
    norm_flops = np.prod(inp.shape).item()
    if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and module.affine:
        norm_flops *= 2
    if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
        norm_flops *= 2
    return norm_flops


def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.GELU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size
    for s in inp.size()[1:]:
        active_elements_count *= s
    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape).item()


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    batch_size = np.prod(inp.size()[:-1]).item()
    return batch_size * inp.size()[-1] * out.size()[-1]


def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for s in output_size.shape[1:]:
        output_elements_count *= s
    return output_elements_count


if __name__ == "__main__":
    main()

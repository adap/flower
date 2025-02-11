import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import datasets
from config import update_cfg, cfg
from data import fetch_dataset, make_data_loader, make_transform
from utils import save, process_control, process_dataset, collate, Stats, makedir_exist_ok
import hydra
from omegaconf import DictConfig, OmegaConf

stats_path = './res/stats'
dim = 1

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
    process_control()
    assert cfg == globals()['cfg'] 
    cfg['seed'] = 0
    data_names = ['SpeechCommandsV1', 'SpeechCommandsV2']
    with torch.no_grad():
        for data_name in data_names:
            cfg['data_name'] = data_name
            root = os.path.join('data', cfg['data_name'])
            dataset = eval('datasets.{}(root=root, split=\'train\')'.format(cfg['data_name']))
            process_dataset({'train': dataset})
            cfg['data_length'] = 1 * dataset.sr
            cfg['n_fft'] = round(0.04 * dataset.sr)
            cfg['hop_length'] = round(0.02 * dataset.sr)
            cfg['background_noise'] = dataset.background_noise
            plain_transform = make_transform('plain')
            dataset.transform = datasets.Compose([plain_transform])
            data_loader = make_data_loader({'train': dataset}, cfg['model_name'])
            stats = Stats(dim=dim)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                stats.update(input['data'])
            stats = (stats.mean.tolist(), stats.std.tolist())
            print(cfg['data_name'], stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}.pt'.format(cfg['data_name'])))


if __name__ == "__main__":
    main()
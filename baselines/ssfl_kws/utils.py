import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import update_cfg, cfg

def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['data_size'] = {k: len(dataset[k]) for k in dataset}
    cfg['target_size'] = dataset['train'].target_size
    return


def process_control():
    print("in utils\n", cfg)
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    if cfg['control']['num_supervised'] == 'fs':
        cfg['control']['num_supervised'] = '-1'
    cfg['num_supervised'] = int(cfg['control']['num_supervised'])
    aug_list = cfg['control']['aug'].split('=')
    cfg['sup_aug'] = aug_list[0]
    if len(aug_list) > 1:
        cfg['unsup_aug'] = aug_list[1]
    if 'loss_mode' in cfg['control']:
        cfg['loss_mode'] = cfg['control']['loss_mode']
    data_shape = {'SpeechCommandsV1': [1, 40, 51], 'SpeechCommandsV2': [1, 40, 51]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['dscnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['lstm'] = {'hidden_size': 128, 'num_layers': 2}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['tcresnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['tcresnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 37, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['mhattrnn'] = {'hidden_size': 256, 'num_heads': 4, 'dropout': 0.1}
    cfg['threshold'] = 0.99
    cfg['unsup_ratio'] = 1
    cfg['alpha'] = 0.75
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    if model_name in ['lstm', 'mhattrnn']:
        cfg[model_name]['optimizer_name'] = 'Adam'
        cfg[model_name]['lr'] = 1e-3
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['scheduler_name'] = 'None'
        cfg[model_name]['betas'] = (0.9, 0.999)
    elif model_name in ['cnn', 'dscnn']:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['lr'] = 1e-1
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['nesterov'] = True
        cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    else:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['lr'] = 3e-2
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['nesterov'] = True
        cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    cfg[model_name]['num_epochs'] = 2 # 400
    cfg[model_name]['batch_size'] = {'train': 1000, 'test': 1000} # {'train': 250, 'test': 250}
    if 'num_clients' in cfg['control']:
        cfg['num_clients'] = int(cfg['control']['num_clients'])
        cfg['active_rate'] = float(cfg['control']['active_rate'])
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        cfg['gm'] = 0
        cfg['server'] = {}
        cfg['server']['shuffle'] = {'train': True, 'test': False}
        if cfg['num_supervised'] > 1000:
            cfg['server']['batch_size'] = {'train': 1000, 'test': 1000} # {'train': 250, 'test': 500}
        else:
            cfg['server']['batch_size'] = {'train': 100, 'test': 1000} # {'train': 25, 'test': 500}
        cfg['server']['num_epochs'] = 3 # 5
        cfg['client'] = {}
        cfg['client']['shuffle'] = {'train': True, 'test': False}
        cfg['client']['batch_size'] = {'train': 250, 'test': 1000} # {'train': 10, 'test': 500}
        cfg['local'] = {}
        cfg['local']['optimizer_name'] = 'SGD'
        cfg['local']['lr'] = 3e-2
        cfg['local']['momentum'] = 0.9
        cfg['local']['weight_decay'] = 5e-4
        cfg['local']['nesterov'] = True
        cfg['local']['num_epochs'] = 3 #5
        cfg['global'] = {}
        cfg['global']['batch_size'] = {'train': 1000, 'test': 1000} # {'train': 250, 'test': 500}
        cfg['global']['shuffle'] = {'train': True, 'test': False}
        cfg['global']['num_epochs'] = 4 # 800
        cfg['global']['optimizer_name'] = 'SGD'
        cfg['global']['lr'] = 1
        cfg['global']['momentum'] = cfg['gm']
        cfg['global']['weight_decay'] = 0
        cfg['global']['nesterov'] = False
        cfg['global']['scheduler_name'] = 'CosineAnnealingLR'
    torch.set_num_threads(1)
    cfg['stats'] = make_stats()
    # update_cfg(cfg)
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        data_name = os.path.splitext(filename)[0]
        file_path = os.path.join(stats_path, filename)
        stats[data_name] = load(file_path)
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True, resume_mode=1):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)) and resume_mode == 1:
        result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        if verbose:
            print('Resume from {}'.format(result['epoch']))
    else:
        if resume_mode == 1:
            print('Not exists model tag: {}, start from scratch'.format(model_tag))
        result = None
    return result


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input

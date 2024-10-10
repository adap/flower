import copy
import os
import torch
import torchaudio
import torchvision
import numpy as np
import models
import datasets
from config import cfg
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device


def fetch_dataset(data_name, aug='plain'):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['SpeechCommandsV1', 'SpeechCommandsV2']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
        cfg['data_length'] = 1 * dataset['train'].sr
        cfg['n_fft'] = round(0.04 * dataset['train'].sr)
        cfg['hop_length'] = round(0.02 * dataset['train'].sr)
        cfg['background_noise'] = dataset['train'].background_noise
        train_transform = make_transform(aug)
        test_transform = make_transform('plain')
        dataset['train'].transform = datasets.Compose(
            [train_transform, torchvision.transforms.Normalize(*cfg['stats'][data_name])])
        dataset['test'].transform = datasets.Compose(
            [test_transform, torchvision.transforms.Normalize(*cfg['stats'][data_name])])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=input_collate, worker_init_fn=np.random.seed(cfg['seed']))
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=input_collate, worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=input_collate, worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader


def split_dataset(dataset, num_splits, data_split_mode):
    if data_split_mode == 'iid':
        data_split, target_split = iid(dataset, num_splits)
    elif 'non-iid' in data_split_mode:
        data_split, target_split = non_iid(dataset, num_splits)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, target_split


def iid(dataset, num_splits):
    data_split = [{k: None for k in dataset} for _ in range(num_splits)]
    target_split = [{k: None for k in dataset} for _ in range(num_splits)]
    for k in dataset:
        idx_k = torch.randperm(len(dataset[k]))
        data_split_k = torch.tensor_split(idx_k, num_splits)
        for i in range(num_splits):
            data_split[i][k] = data_split_k[i].tolist()
            target_i_k = torch.tensor(dataset[k].target)[data_split[i][k]]
            if k == 'train':
                unique_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                target_split[i][k] = {unique_target_i_k[m].item(): num_target_i[m].item()
                                      for m in range(len(unique_target_i_k))}
            else:
                target_split[i][k] = {x: int((target_i_k == x).sum()) for x in target_split[i]['train']}
    return data_split, target_split


def non_iid(dataset, num_splits):
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    target_size = len(torch.unique(torch.tensor(dataset['train'].target)))
    if data_split_mode_tag == 'l':
        data_split = [{k: [] for k in dataset} for _ in range(num_splits)]
        shard_per_user = int(data_split_mode_list[-1])
        shard_per_class = int(np.ceil(shard_per_user * num_splits / target_size))
        target_idx_split = [{k: None for k in dataset} for _ in range(target_size)]
        for k in dataset:
            target = torch.tensor(dataset[k].target)
            for target_i in range(target_size):
                target_idx = torch.where(target == target_i)[0]
                num_leftover = len(target_idx) % shard_per_class
                leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
                target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
                target_idx = target_idx.reshape((shard_per_class, -1)).tolist()
                for i, leftover_target_idx in enumerate(leftover):
                    target_idx[i].append(leftover_target_idx.item())
                target_idx_split[target_i][k] = target_idx
        target_split_key = []
        for i in range(shard_per_class):
            target_split_key.append(torch.randperm(target_size))
        target_split_key = torch.cat(target_split_key, dim=0)
        target_split = [{k: None for k in dataset} for _ in range(num_splits)]
        exact_size = shard_per_user * num_splits
        exact_target_split, leftover_target_split = target_split_key[:exact_size].tolist(), \
                                                    {k: target_split_key[exact_size:].tolist() for k in dataset}
        for i in range(0, exact_size, shard_per_user):
            target_split_i = exact_target_split[i:i + shard_per_user]
            for j in range(len(target_split_i)):
                target_i_j = target_split_i[j]
                for k in dataset:
                    idx = torch.randint(len(target_idx_split[target_i_j][k]), (1,)).item()
                    data_split[i // shard_per_user][k].extend(target_idx_split[target_i_j][k].pop(idx))
                    if target_i_j in leftover_target_split[k]:
                        idx = torch.randint(len(target_idx_split[target_i_j][k]), (1,)).item()
                        data_split[i // shard_per_user][k].extend(target_idx_split[target_i_j][k].pop(idx))
                        leftover_idx = leftover_target_split[k].index(target_i_j)
                        leftover_target_split[k].pop(leftover_idx)
                    target_i_j_k = torch.tensor(dataset[k].target)[data_split[i // shard_per_user][k]]
                    if k == 'train':
                        unique_target_i_k, num_target_i = torch.unique(target_i_j_k, sorted=True, return_counts=True)
                        target_split[i // shard_per_user][k] = {unique_target_i_k[m].item(): num_target_i[m].item()
                                                                for m in range(len(unique_target_i_k))}
                    else:
                        target_split[i // shard_per_user][k] = {x: int((target_i_j_k == x).sum()) for x in
                                                                target_split[i // shard_per_user]['train']}
    elif data_split_mode_tag == 'd':
        data_split, target_split = None, None
        min_size = 0
        required_min_size = 10
        while min_size < required_min_size:
            beta = float(data_split_mode_list[-1])
            dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_splits))
            data_split = [{k: [] for k in dataset} for _ in range(num_splits)]
            for target_i in range(target_size):
                proportions = dir.sample()
                for k in dataset:
                    target = torch.tensor(dataset[k].target)
                    target_idx = torch.where(target == target_i)[0]
                    proportions = torch.tensor([p * (len(data_split_idx[k]) < (len(target) / num_splits))
                                                for p, data_split_idx in zip(proportions, data_split)])
                    proportions = proportions / proportions.sum()
                    split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                    split_idx = torch.tensor_split(target_idx, split_idx)
                    for i in range(len(split_idx)):
                        data_split[i][k].extend(split_idx[i].tolist())
            min_size = min([len(data_split[i]['train']) for i in range(len(data_split))])
            target_split = [{k: None for k in dataset} for _ in range(num_splits)]
            for i in range(num_splits):
                for k in dataset:
                    target_i_k = torch.tensor(dataset[k].target)[data_split[i][k]]
                    if k == 'train':
                        unique_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                        target_split[i][k] = {unique_target_i_k[m].item(): num_target_i[m].item() for m in
                                              range(len(unique_target_i_k))}
                    else:
                        target_split[i][k] = {x: (target_i_k == x).sum().item() for x in target_split[i]['train']}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split, target_split


def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.id = list(range(len(separated_dataset.data)))
    return separated_dataset


def separate_dataset_semi(dataset, supervised_idx=None):
    if supervised_idx is None:
        if cfg['num_supervised'] == -1:
            supervised_idx = list(range(len(dataset)))
        else:
            target = torch.tensor(dataset.target)
            num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
            supervised_idx = []
            for i in range(cfg['target_size']):
                idx = torch.where(target == i)[0]
                idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                supervised_idx.extend(idx)
    idx = list(range(len(dataset)))
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    sup_dataset = separate_dataset(dataset, supervised_idx)
    unsup_dataset = separate_dataset(dataset, unsupervised_idx)
    return sup_dataset, unsup_dataset, supervised_idx


def make_batchnorm_dataset(dataset):
    dataset = copy.deepcopy(dataset)
    plain_transform = datasets.Compose(
        [make_transform('plain'), torchvision.transforms.Normalize(*cfg['stats'][cfg['data_name']])])
    dataset.transform = plain_transform
    return dataset


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def make_transform(mode):
    if mode == 'plain':
        transform = make_plain_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic':
        transform = make_basic_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-spec':
        transform = make_basic_spec_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-rand':
        transform = make_basic_rand_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-rands':
        transform = make_basic_rands_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-spec-rands':
        transform = make_basic_spec_rands_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif 'fix' in mode:
        transform = make_fix_transform(cfg['data_name'])
    else:
        raise ValueError('Not valid aug')
    return transform


def make_fix_transform(data_name):
    transform = FixTransform(data_name)
    return transform


def make_plain_transform(data_length, n_fft, hop_length):
    plain_transform = [datasets.transforms.CenterCropPad(data_length),
                       torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                       torchaudio.transforms.AmplitudeToDB('power', 80),
                       datasets.transforms.SpectoImage(),
                       torchvision.transforms.ToTensor()]
    plain_transform = torchvision.transforms.Compose(plain_transform)
    return plain_transform


def make_basic_transform(data_length, n_fft, hop_length):
    basic_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                       datasets.transforms.CenterCropPad(data_length),
                       datasets.transforms.RandomTimeShift(0.1),
                       torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                       torchaudio.transforms.AmplitudeToDB('power', 80),
                       datasets.transforms.SpectoImage(),
                       torchvision.transforms.ToTensor()]
    basic_transform = torchvision.transforms.Compose(basic_transform)
    return basic_transform


def make_basic_spec_transform(data_length, n_fft, hop_length):
    basic_spec_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                            datasets.transforms.CenterCropPad(data_length),
                            datasets.transforms.RandomTimeShift(0.1),
                            torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                            torchaudio.transforms.FrequencyMasking(7),
                            torchaudio.transforms.TimeMasking(12),
                            torchaudio.transforms.AmplitudeToDB('power', 80),
                            datasets.transforms.SpectoImage(),
                            torchvision.transforms.ToTensor()]
    basic_spec_transform = torchvision.transforms.Compose(basic_spec_transform)
    return basic_spec_transform


def make_basic_rand_transform(data_length, n_fft, hop_length):
    basic_rand_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                            datasets.transforms.CenterCropPad(data_length),
                            datasets.transforms.RandomTimeShift(0.1),
                            torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                            torchaudio.transforms.AmplitudeToDB('power', 80),
                            datasets.transforms.SpectoImage(),
                            datasets.randaugment.RandAugment(n=2, m=10),
                            torchvision.transforms.ToTensor()]
    basic_rand_transform = torchvision.transforms.Compose(basic_rand_transform)
    return basic_rand_transform


def make_basic_rands_transform(data_length, n_fft, hop_length):
    basic_rands_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                             datasets.transforms.CenterCropPad(data_length),
                             datasets.transforms.RandomTimeShift(0.1),
                             torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                             torchaudio.transforms.AmplitudeToDB('power', 80),
                             datasets.transforms.SpectoImage(),
                             datasets.randaugment.RandAugmentSelected(n=2, m=10),
                             torchvision.transforms.ToTensor()]
    basic_rands_transform = torchvision.transforms.Compose(basic_rands_transform)
    return basic_rands_transform


def make_basic_spec_rands_transform(data_length, n_fft, hop_length):
    basic_spec_rands_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                                  datasets.transforms.CenterCropPad(data_length),
                                  datasets.transforms.RandomTimeShift(0.1),
                                  torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                                  torchaudio.transforms.FrequencyMasking(7),
                                  torchaudio.transforms.TimeMasking(12),
                                  torchaudio.transforms.AmplitudeToDB('power', 80),
                                  datasets.transforms.SpectoImage(),
                                  datasets.randaugment.RandAugmentSelected(n=2, m=10),
                                  torchvision.transforms.ToTensor()]
    basic_spec_rands_transform = torchvision.transforms.Compose(basic_spec_rands_transform)
    return basic_spec_rands_transform


class FixTransform(torch.nn.Module):
    def __init__(self, data_name):
        super().__init__()
        self.weak = datasets.Compose(
            [make_transform(cfg['sup_aug']), torchvision.transforms.Normalize(*cfg['stats'][data_name])])
        self.strong = datasets.Compose(
            [make_transform(cfg['unsup_aug']), torchvision.transforms.Normalize(*cfg['stats'][data_name])])

    def forward(self, input):
        data = self.weak({'data': input['data']})['data']
        aug_data = self.strong({'data': input['data']})['data']
        input = {**input, 'data': data, 'aug_data': aug_data}
        return input


class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        input = {'data': input['data'], 'target': input['target']}
        return input

    def __len__(self):
        return self.size

import argparse
import copy
import datetime
import models
import os
import shutil
import time
import numpy as np
import datasets
import torch
import torch.backends.cudnn as cudnn
from config import update_cfg, cfg
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset_semi, make_transform, \
    make_batchnorm_dataset, make_batchnorm_stats
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
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
    # pprint(cfg)
    process_control()
    assert cfg == globals()['cfg']

    # print("globals", globals()['cfg'])    
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return

def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'], cfg['sup_aug'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    sup_dataset, unsup_dataset, supervised_idx = separate_dataset_semi(dataset['train'])
    unsup_dataset.transform = make_transform(cfg['loss_mode'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                     'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], resume_mode=cfg['resume_mode'])
    if result is None:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = result['epoch']
        supervised_idx = result['supervised_idx']
        model.load_state_dict(result['model_state_dict'])
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        logger = result['logger']
        sup_dataset, unsup_dataset, supervised_idx = separate_dataset_semi(dataset['train'], supervised_idx)
        unsup_dataset.transform = make_transform(cfg['loss_mode'])
    unsup_dataloader = make_data_loader({'train': unsup_dataset}, cfg['model_name'],
                                        batch_size={'train': cfg[cfg['model_name']]['batch_size']['train'] * cfg[
                                            'unsup_ratio']})
    sup_sampler = SupSampler(len(unsup_dataloader['train']), cfg[cfg['model_name']]['batch_size']['train'],
                             len(sup_dataset))
    sup_dataloader = make_data_loader({'train': sup_dataset}, cfg['model_name'], batch_sampler={'train': sup_sampler})
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        train(sup_dataloader['train'], unsup_dataloader['train'], model, optimizer, metric, logger, epoch)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, cfg['model_name'])
        test(data_loader['test'], test_model, metric, logger, epoch)
        scheduler.step()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train(sup_dataloader, unsup_dataloader, model, optimizer, metric, logger, epoch):
    logger.safe(True)
    model.train(True)
    start_time = time.time()
    beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
    for i, (sup_input, unsup_input) in enumerate(zip(sup_dataloader, unsup_dataloader)):
        # Sup
        sup_input = collate(sup_input)
        sup_input = to_device(sup_input, cfg['device'])
        # Fix
        unsup_input = collate(unsup_input)
        unsup_input = to_device(unsup_input, cfg['device'])
        with torch.no_grad():
            output_, input_ = {}, {}
            model.train(False)
            unsup_output = model({'data': unsup_input['data']})
            input_['target'] = unsup_input['target']
            output_['target'] = torch.softmax(unsup_output['target'], dim=-1)
            new_target, mask = make_hard_pseudo_label(output_['target'])
            output_['mask'] = mask
            evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
            logger.append(evaluation, 'train', n=len(input_['target']))
            unsup_input['target'] = new_target.detach()
        if torch.any(mask):
            unsup_input['data'] = unsup_input['data'][mask]
            unsup_input['aug_data'] = unsup_input['aug_data'][mask]
            unsup_input['target'] = unsup_input['target'][mask]
            # Mix
            if 'mix' in cfg['loss_mode']:
                mix_size = min(len(sup_input['data']), len(unsup_input['data']))
                lam = beta.sample()[0]
                unsup_input['mix_data'] = (lam * sup_input['data'][:mix_size] +
                                           (1 - lam) * unsup_input['data'][:mix_size]).detach()
                unsup_input['mix_target'] = torch.stack([sup_input['target'][:mix_size],
                                                         unsup_input['target'][:mix_size]], dim=-1).detach()
                input = {'data': sup_input['data'], 'target': sup_input['target'], 'aug_data': unsup_input['aug_data'],
                         'aug_target': unsup_input['target'], 'mix_data': unsup_input['mix_data'],
                         'mix_target': unsup_input['mix_target'], 'lam': lam}
            else:
                input = {'data': sup_input['data'], 'target': sup_input['target'], 'aug_data': unsup_input['aug_data'],
                         'aug_target': unsup_input['target']}
        else:
            input = {'data': sup_input['data'], 'target': sup_input['target']}
        model.train(True)
        input_size = input['data'].size(0)
        input['loss_mode'] = cfg['loss_mode']
        output = model(input)
        optimizer.zero_grad()
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(['Accuracy', 'Loss'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(sup_dataloader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(sup_dataloader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(sup_dataloader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(sup_dataloader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


class SupSampler(torch.utils.data.Sampler):
    def __init__(self, num_batches, batch_size, data_size):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data_size = data_size

    def __iter__(self):
        total_size = self.num_batches * self.batch_size
        idx = []
        for i in range(int(np.ceil(total_size / self.data_size))):
            idx_i = torch.randperm(self.data_size)
            idx.append(idx_i)
        idx = torch.cat(idx, dim=0)[:total_size]
        idx = torch.chunk(idx, self.num_batches)
        yield from idx

    def __len__(self):
        return self.num_batches


def make_hard_pseudo_label(soft_pseudo_label):
    max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
    mask = max_p.ge(cfg['threshold'])
    return hard_pseudo_label, mask


if __name__ == "__main__":
    main()

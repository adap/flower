import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import update_cfg, cfg
from data import fetch_dataset, make_data_loader, separate_dataset_semi, make_batchnorm_dataset, make_batchnorm_stats
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
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
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    batchnorm_dataset = make_batchnorm_dataset(dataset['train'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
    metric = Metric({'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'], load_tag='best', resume_mode=1)
    last_epoch = result['epoch']
    supervised_idx = result['supervised_idx']
    model.load_state_dict(result['model_state_dict'])
    dataset['train'], _, supervised_idx = separate_dataset_semi(dataset['train'], supervised_idx=supervised_idx)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_model = make_batchnorm_stats(batchnorm_dataset, model, cfg['model_name'])
    test(data_loader['test'], test_model, metric, test_logger, last_epoch)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    result = {'cfg': cfg, 'epoch': last_epoch, 'supervised_idx': supervised_idx,
              'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
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


if __name__ == "__main__":
    main()

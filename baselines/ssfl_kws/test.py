import datasets
from config import cfg
from data import fetch_dataset, make_data_loader, make_transform
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import torchvision
import torchaudio
import models

if __name__ == "__main__":
    cfg['seed'] = 0
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['control']['data_name'] = 'SpeechCommandsV1'
    process_control()
    aug = 'plain'
    dataset = fetch_dataset(cfg['data_name'])
    print(len(dataset['train']))
    print(len(dataset['test']))
    exit()
    dataset['train'].transform = datasets.Compose([make_transform(aug)])
    print(dataset['train'].transform)
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'], shuffle={'train': False, 'test': False})
    print(len(dataset['train']), len(dataset['test']))
    print(len(data_loader['train']), len(data_loader['test']))
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        print(i, input['data'].shape, input['target'].shape)
        torchvision.utils.save_image(input['data'], './output/train.png')
        # torchaudio.save('./output/temp.wav', input['data'][0], 16000)
        break
    exit()
    for i, input in enumerate(data_loader['test']):
        input = collate(input)
        print(i, input['data'].shape, input['target'].shape)
        torchvision.utils.save_image(input['data'], './output/test.png')
        # break

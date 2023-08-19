import os
import numpy as np
import time
import argparse
import sys

import math
from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models

from distoptim import FedProx, FedNova
import util_v4 as util
import logging
import time

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--name','-n', 
                    default="default", 
                    type=str, 
                    help='experiment name, used for saving results')
parser.add_argument('--backend',
                    default="nccl",
                    type=str,
                    help='background name')
parser.add_argument('--model', 
                    default="res", 
                    type=str, 
                    help='neural network model')
parser.add_argument('--alpha', 
                    default=0.2, 
                    type=float, 
                    help='control the non-iidness of dataset')
parser.add_argument('--gmf', 
                    default=0, 
                    type=float, 
                    help='global (server) momentum factor')
parser.add_argument('--lr', 
                    default=0.1, 
                    type=float, 
                    help='client learning rate')
parser.add_argument('--momentum', 
                    default=0.0, 
                    type=float, 
                    help='local (client) momentum factor')
parser.add_argument('--bs', 
                    default=512, 
                    type=int, 
                    help='batch size on each worker/client')
parser.add_argument('--rounds', 
                    default=200, 
                    type=int, 
                    help='total coommunication rounds')
parser.add_argument('--localE', 
                    default=98, 
                    type=int, 
                    help='number of local epochs')
parser.add_argument('--print_freq', 
                    default=100, 
                    type=int, 
                    help='print info frequency')
parser.add_argument('--size', 
                    default=8, 
                    type=int, 
                    help='number of local workers')
parser.add_argument('--rank', 
                    default=0, 
                    type=int, 
                    help='the rank of worker')
parser.add_argument('--seed', 
                    default=1, 
                    type=int, 
                    help='random seed')
parser.add_argument('--save', '-s', 
                    action='store_true', 
                    help='whether save the training results')
parser.add_argument('--p', '-p', 
                    action='store_true', 
                    help='whether the dataset is partitioned or not')
parser.add_argument('--NIID',
                    action='store_true',
                    help='whether the dataset is non-iid or not')
parser.add_argument('--pattern',
                    type=str, 
                    help='pattern of local steps')
parser.add_argument('--optimizer', 
                    default='local', 
                    type=str, 
                    help='optimizer name')
parser.add_argument('--initmethod',
                    default='tcp://h0:22000',
                    type=str,
                    help='init method')
parser.add_argument('--mu', 
                    default=0, 
                    type=float, 
                    help='mu parameter in fedprox')
parser.add_argument('--savepath',
                    default='./results/',
                    type=str,
                    help='directory to save exp results')
parser.add_argument('--datapath',
                    default='./data/',
                    type=str,
                    help='directory to load data')


logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logging.debug('This message should appear on the console')
args = parser.parse_args()

def run(rank, size):
    # initiate experiments folder
    save_path = args.savepath
    folder_name = save_path+args.name
    if rank == 0 and os.path.isdir(folder_name)==False and args.save:
        os.mkdir(folder_name)
    dist.barrier()

    # initiate log files
    tag = '{}/lr{:.3f}_bs{:d}_cp{:d}_a{:.2f}_e{}_r{}_n{}.csv'
    saveFileName = tag.format(folder_name, args.lr, args.bs, args.localE, 
                              args.alpha, args.seed, rank, size)
    args.out_fname = saveFileName
    with open(args.out_fname, 'w+') as f:
        print(
            'BEGIN-TRAINING\n'
            'World-Size,{ws}\n'
            'Batch-Size,{bs}\n'
            'Epoch,itr,'
            'Loss,avg:Loss,Prec@1,avg:Prec@1,val,time'.format(
                ws=args.size,
                bs=args.bs),
            file=f)


    # seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load datasets
    train_loader, test_loader, DataRatios = \
        util.partition_dataset(rank, size, args)
    logging.debug("Worker id {} local sample ratio {} "
                  "local epoch length {}"
                  .format(rank, DataRatios[rank], len(train_loader)))

    # define neural nets model, criterion, and optimizer
    model = util.select_model(10, args).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # select optimizer according to algorithm
    algorithms = {
        'fedavg': FedProx, # mu = 0
        'fedprox': FedProx,
        # 'scaffold': Scaffold,
        'fednova': FedNova,
        # 'fednova_vr':FedNovaVR,
    }
    selected_opt = algorithms[args.optimizer]
    optimizer = selected_opt(model.parameters(),
                             lr=args.lr,
                             gmf=args.gmf,
                             mu=args.mu,
                             ratio=DataRatios[rank],
                             momentum=args.momentum,
                             nesterov = False,
                             weight_decay=1e-4)

    best_test_accuracy = 0
    for rnd in range(args.rounds):
        # Decide number of local updates per client
        local_epochs = update_local_epochs(args.pattern, rank, rnd)
        tau_i = local_epochs * len(train_loader)
        logging.info("local epochs {} iterations {}"
                      .format(local_epochs, tau_i))

        # Decay learning rate according to round index
        update_learning_rate(optimizer, rnd, args.lr)
        # Clients locally train for several local epochs
        for t in range(local_epochs):
            train(model, criterion, optimizer, train_loader, t)
        
        # synchronize parameters
        dist.barrier()
        comm_start = time.time()
        optimizer.average()
        dist.barrier()
        comm_end = time.time()
        comm_time = comm_end - comm_start
        
        # evaluate test accuracy
        test_acc = evaluate(model, test_loader)
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
        
        # record metrics
        logging.info("Round {} test accuracy {:.3f} time {:.3f}".format(rnd, test_acc, comm_time))
        with open(args.out_fname, '+a') as f:
            print('{ep},{itr},{filler},{filler},'
                  '{filler},{filler},'
                  '{val:.4f},{time:.4f}'
                  .format(ep=rnd, itr=-1,
                          filler=-1, val=test_acc, time=comm_time), file=f)

    logging.info("Worker {} best test accuracy {:.3f}"
                 .format(rank, best_test_accuracy))


def evaluate(model, test_loader):
    model.eval()
    top1 = util.Meter(ptag='Acc')

    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)
            outputs = model(data)
            acc1 = util.comp_accuracy(outputs, target)
            top1.update(acc1[0].item(), data.size(0))

    return top1.avg


def train(model, criterion, optimizer, loader, epoch):

    model.train()

    losses = util.Meter(ptag='Loss')
    top1 = util.Meter(ptag='Prec@1')

    for batch_idx, (data, target) in enumerate(loader):
        # data loading
        data = data.cuda(non_blocking = True)
        target = target.cuda(non_blocking = True)

        # forward pass
        output = model(data)
        loss = criterion(output, target)

        # backward pass
        loss.backward()

        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        # write log files
        train_acc = util.comp_accuracy(output, target)
        

        losses.update(loss.item(), data.size(0))
        top1.update(train_acc[0].item(), data.size(0))

        if batch_idx % args.print_freq == 0 and args.save:
            logging.debug('epoch {} itr {}, '
                         'rank {}, loss value {:.4f}, train accuracy {:.3f}'
                         .format(epoch, batch_idx, rank, losses.avg, top1.avg))

            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},'
                      '{loss.val:.4f},{loss.avg:.4f},'
                      '{top1.val:.3f},{top1.avg:.3f},-1'
                      .format(ep=epoch, itr=batch_idx,
                              loss=losses, top1=top1), file=f)

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{top1.val:.3f},{top1.avg:.3f},-1'
              .format(ep=epoch, itr=batch_idx,
                      loss=losses, top1=top1), file=f)


def update_local_epochs(pattern, rank, rnd):
    if pattern == "constant":
        return args.localE

    if pattern == "uniform_random":
        np.random.seed(2020+rank+rnd+args.seed)
        return np.random.randint(low=2, high=args.localE, size=1)[0]


def update_learning_rate(optimizer, epoch, target_lr):
    """
    1) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: target_lr is the reference learning rate from which to scale down
    """
    if epoch == int(args.rounds / 2):
        lr = target_lr/10
        logging.info('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == int(args.rounds * 0.75):
        lr = target_lr/100
        logging.info('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend=args.backend, 
                            init_method=args.initmethod, 
                            rank=rank, 
                            world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    rank = args.rank
    size = args.size
    print(rank)
    init_processes(rank, size, run)

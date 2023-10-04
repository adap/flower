# Nicola Dinsdale 2022
# Main script for FedHarmony MICCAI 2022
# Federated unlearning pretraining with the ABIDE data
########################################################################################################################
# Import dependencies
import numpy as np
from models.age_predictor import Encoder, Regressor, DomainPredictor
from datasets.nifit_dataset import nifti_dataset_ABIDE_agepred_domain
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.utils import shuffle
import torch.optim as optim
from utils import Args, EarlyStopping_unlearning
from train_utils_fed import train_fedprox_gaussian_unlearning_4_sites, val_fedprox_gaussian_unlearning_4_sites
import sys
import argparse
from losses.confusion_loss import confusion_loss
from losses.FedProxLoss import FedProxLoss
import json
import copy
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 75
args.batch_size = 16
args.alpha = 100
args.patience = 25
args.train_val_prop = 0.78
args.learning_rate = 1e-4

cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
parser = argparse.ArgumentParser(description='Define Inputs for pruning model')
parser.add_argument('-s', action="store", dest="Site")
parser.add_argument('-i', action="store", dest="Iteration")

results = parser.parse_args()
try:
    site = str(results.Site)
    iteration = int(results.Iteration)

    print('Training Site : ', site)
    print('Current Iteration : ', iteration)
except:
    raise Exception('Arguement not supplied')
LOAD_PATH_ENCODER = 0
if iteration == 1:
#     LOAD_PATH_ENCODER = 'encoder_initialisation'
#     LOAD_PATH_REGRESSOR = 'regressor_iinitialisation'
#     LOAD_PATH_DOMAIN = 'domain_initialisation'
     LOSS_PATH = 'loss_store_' + site
     loss_store = []
     a_mae = 0
     a_std = 0
     b_mae = 0
     b_std = 0
     c_mae = 0
     c_std = 0
     d_mae = 0
     d_std = 0

if iteration > 1:
    LOAD_PATH_ENCODER = 'encoder_aggregated_' + str(iteration - 1)
    LOAD_PATH_REGRESSOR = 'regressor_aggregated_' + str(iteration - 1)
    LOAD_PATH_DOMAIN = 'domain_aggregated_' + str(iteration - 1)
    LOAD_PATH_LOSSES = 'loss_store_' + site + '.npy'
    loss_store = np.load(LOAD_PATH_LOSSES)
    loss_store = np.ndarray.tolist(loss_store)
    LOSS_PATH = LOAD_PATH_LOSSES
    a_mae = np.load('a_mean_' + str(iteration - 1) +'.npy')
    a_std = np.load('a_std_' + str(iteration - 1) +'.npy')
    b_mae = np.load('b_mean_' + str(iteration - 1) +'.npy')
    b_std = np.load('b_std_' + str(iteration - 1) +'.npy')
    c_mae = np.load('c_mean_' + str(iteration - 1) +'.npy')
    c_std = np.load('c_std_' + str(iteration - 1) +'.npy')
    d_mae = np.load('d_mean_' + str(iteration - 1) +'.npy')
    d_std = np.load('d_std_' + str(iteration - 1) +'.npy')

CHK_PATH_ENCODER = 'encoder_checkpoint_' + site + '_' + str(iteration)
CHK_PATH_REGRESSOR = 'regressor_checkpoint_' + site + '_' + str(iteration)
CHK_PATH_DOMAIN = 'domain_checkpoint_' + site + '_' + str(iteration)
#notInclude = ['A00032067','A00032077','A00032121',' A00032086','A00032159','A00032245']
########################################################################################################################
dists = []
site_dict = {'Trinity': 'a', 'NYU':'b', 'UCLA':'c', 'Yale':'d'}
if site == 'Trinity':
    dists = [1, b_mae, b_std, 2, c_mae, c_std, 3, d_mae, d_std]
elif site == 'NYU':
    dists = [0, a_mae, a_std, 2, c_mae, c_std, 3, d_mae, d_std]
elif site == 'UCLA':
    dists = [0, a_mae, a_std, 1, b_mae, b_std, 3, d_mae, d_std]
elif site == 'Yale':
    dists = [0, a_mae, a_std, 1, b_mae, b_std, 2, c_mae, c_std]

########################################################################################################################
train_pths = np.load('train_files_age_pred.npy')
imsize = (160, 240, 160)

with open('age_dictionary_ABIDE.json') as f:
    age_dict = json.load(f)

train_pths_site = []
for i in range(0, len(train_pths)):
    parts = train_pths[i].split('\\')
    
    if parts[6] == site.lower() and parts[13] == 'mprage_0001':
        train_pths_site.append(train_pths[i])
train_pths_site = np.array(train_pths_site)

# count = 0
# import nibabel as nib
# for pth in train_pths_site:
#     data = nib.load(pth).get_fdata()[:, :, 5:165]
#     if (len(data.shape)) == 3:
#         count = count + 1
#     print("data shape:- "+str(data.shape))
# print(count)

train_pths = shuffle(train_pths_site, random_state=0)
proportion = int(args.train_val_prop * len(train_pths))
last_batch = len(train_pths) - args.batch_size
pths_train = train_pths[:144]
pths_val = train_pths[144:]

print('Data splits')
print(train_pths.shape)
print(pths_train.shape, pths_val.shape)

print('Creating datasets and dataloaders')
train_dataset = nifti_dataset_ABIDE_agepred_domain(pths_train, age_dict, site_dict[site])
val_dataset = nifti_dataset_ABIDE_agepred_domain(pths_val, age_dict, site_dict[site])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

# Load the model
encoder = Encoder()
regressor = Regressor()
domain_predictor = DomainPredictor(4)

if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()
    domain_predictor = domain_predictor.cuda()

if iteration>1:
    if LOAD_PATH_ENCODER:
        print('Loading Weights')
        encoder_dict = encoder.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_ENCODER)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
        encoder.load_state_dict(torch.load(LOAD_PATH_ENCODER))
    if LOAD_PATH_REGRESSOR:
        print('Loading Weights')
        encoder_dict = regressor.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_REGRESSOR)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded regressor = ', len(pretrained_dict), '/', len(encoder_dict))
        regressor.load_state_dict(torch.load(LOAD_PATH_REGRESSOR))
    if LOAD_PATH_DOMAIN:
        print('Loading Weights')
        encoder_dict = domain_predictor.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_DOMAIN)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded domain predictor = ', len(pretrained_dict), '/', len(encoder_dict))
        domain_predictor.load_state_dict(torch.load(LOAD_PATH_DOMAIN))

encoder_global = copy.deepcopy(encoder.state_dict())
regressor_global = copy.deepcopy(regressor.state_dict())
criterion = FedProxLoss([encoder_global, regressor_global], mu=0.1)
domain_criterion = nn.CrossEntropyLoss()
conf_criterion = confusion_loss()
if cuda:
    criterion = criterion.cuda()
    domain_criterion = domain_criterion.cuda()
    conf_criterion = conf_criterion.cuda()

optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-4)
optimizer_conf = optim.Adam(list(encoder.parameters()), lr=0.5e-4)
optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-4)

# Initalise the early stopping
early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)
loss_store = []

models = [encoder, regressor, domain_predictor]
optimizers = [optimizer, optimizer_conf, optimizer_dm]
criterions = [criterion, conf_criterion, domain_criterion]

epoch_reached = 1

for epoch in range(epoch_reached, args.epochs+1):
    print('Epoch ', epoch, '/', args.epochs, flush=True)
    loss, acc, dm_loss, conf_loss = train_fedprox_gaussian_unlearning_4_sites(args, models, train_dataloader, optimizers, criterions, epoch, dists)

    val_loss, val_acc = val_fedprox_gaussian_unlearning_4_sites(args, models, val_dataloader, criterions, dists)
    loss_store.append([loss, acc, dm_loss, conf_loss, val_loss, val_acc])
    np.save(LOSS_PATH, np.array(loss_store))

    # Decide whether the model should stop training or not
    early_stopping(val_loss, models, epoch, optimizer, loss, [CHK_PATH_ENCODER, CHK_PATH_REGRESSOR, CHK_PATH_DOMAIN])

    if early_stopping.early_stop:
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)
        sys.exit('Patience Reached - Early Stopping Activated')

    if epoch == args.epochs:
        print('Finished Training', flush=True)
        loss_store = np.array(loss_store)
        np.save(LOSS_PATH, loss_store)

    torch.cuda.empty_cache()  # Clear memory cache


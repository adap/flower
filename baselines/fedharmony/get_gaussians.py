# Nicola Dinsdale 2022
# Get the mean and stds to share
########################################################################################################################
from models.age_predictor import Regressor, Encoder, DomainPredictor
from datasets.nifti_dataset import nifti_dataset_ABIDE_agepred_domain
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from utils import Args
from losses.confusion_loss import confusion_loss
from losses.DANN_loss import DANN_loss
import torch.optim as optim
import json
from sklearn.utils import shuffle
import argparse
from torch.autograd import Variable

########################################################################################################################
def get_embeddings(args, models, criterion, test_loader):
    cuda = torch.cuda.is_available()
    embeddings = []

    [encoder, regressor, domain_predictor] = models
    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    with torch.no_grad():
        for data, target, domain_target in test_loader:
            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()
            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)
            if list(data.size())[0] == args.batch_size:
                features = encoder(data)
                embeddings.append(features.detach().cpu().numpy())

    embeddings = np.array(embeddings)
    return embeddings
########################################################################################################################
parser = argparse.ArgumentParser(description='Define Inputs for pruning model')
parser.add_argument('-i', action="store", dest="Iteration")
results = parser.parse_args()
try:
    iteration = int(results.Iteration)
    print('Current Iteration : ', iteration)
except:
    raise Exception('Arguement not supplied')
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 200
args.batch_size = 2
args.alpha = 1
args.patience = 50
args.learning_rate = 1e-4

LOAD_PATH_ENCODER = 'encoder_aggregated_' + str(iteration)
LOAD_PATH_REGRESSOR = 'regressor_aggregated_' + str(iteration)
LOAD_PATH_DOMAIN = 'domain_aggregated_' + str(iteration)

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda Available', flush=True)
########################################################################################################################
site_dict = {'Trinity': 'a', 'NYU':'b', 'UCLA':'c', 'Yale':'d'}

for site in  ['Trinity', 'NYU', 'UCLA', 'Yale']:
    train_pths = np.load('train_files_age_pred.npy')
    imsize = (160, 240, 160)

    with open('age_dictionary_ABIDE.json') as f:
        age_dict = json.load(f)

    train_pths_site = []
    print(site, flush=True)
    for i in range(0, len(train_pths)):
        parts = train_pths[i].split('/')
        if parts[6] == site:
            train_pths_site.append(train_pths[i])
    train_pths_site = np.array(train_pths_site)
    print(train_pths_site)
    train_pths = shuffle(train_pths_site, random_state=0)  # Same seed everytime
    proportion = int(args.train_val_prop * len(train_pths))
    last_batch = len(train_pths) - args.batch_size
    pths_train = train_pths[:last_batch]
    print(pths_train)
    pths_val = train_pths[last_batch:]
    train_dataset = nifti_dataset_ABIDE_agepred_domain(pths_train, age_dict, site_dict[site])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load the model
    encoder = Encoder()
    regressor = Regressor()
    domain_predictor = DomainPredictor(4)
    if cuda:
        encoder = encoder.cuda()
        regressor = regressor.cuda()
        domain_predictor = domain_predictor.cuda()

    if LOAD_PATH_ENCODER:
        print('Loading Weights')
        encoder_dict = encoder.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_ENCODER)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
        encoder.load_state_dict(torch.load(LOAD_PATH_ENCODER))

    if LOAD_PATH_REGRESSOR:
        regressor_dict = regressor.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_REGRESSOR)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
        print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))
        regressor.load_state_dict(torch.load(LOAD_PATH_REGRESSOR))

    if LOAD_PATH_DOMAIN:
        domain_dict = domain_predictor.state_dict()
        pretrained_dict = torch.load(LOAD_PATH_DOMAIN)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in domain_dict}
        print('weights loaded domain predictor = ', len(pretrained_dict), '/', len(domain_dict))
        domain_predictor.load_state_dict(torch.load(LOAD_PATH_DOMAIN))

    criteron = DANN_loss()
    criteron.cuda()
    domain_criterion = nn.BCELoss()
    domain_criterion.cuda()
    conf_criterion = confusion_loss()
    conf_criterion.cuda()

    optimizer_step1 = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()) + list(domain_predictor.parameters()), lr=args.learning_rate)
    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-5)
    optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-5)
    optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-5)

    models = [encoder, regressor, domain_predictor]

    print('Mean ' + site)
    embeddings= get_embeddings(args, models, criteron, train_dataloader)
    embeddings = np.reshape(embeddings, (-1, 64))
    print(embeddings.shape)
    print(embeddings)

    noise = np.random.normal(size=embeddings.shape) * 1e-8
    print(noise.shape)
    embeddings = embeddings + noise
    b_mean = np.mean(embeddings, axis=0)
    print(b_mean.shape)
    b_std = np.std(embeddings, axis=0)
    print(b_std.shape)

    letter = site_dict[site]
    print(letter)
    np.save(str(letter)+'_mean_' + str(iteration), b_mean)
    np.save(str(letter)+'_std_' + str(iteration), b_std)

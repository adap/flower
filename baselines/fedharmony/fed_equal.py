# Nicola Dinsdale 2022
# Fed Avg but with each site contributing equally
########################################################################################################################
# Import dependencies
import numpy as np
from models.age_predictor import Encoder, Regressor, DomainPredictor
import argparse
import torch
from collections import OrderedDict

cuda = torch.cuda.is_available()

########################################################################################################################
parser = argparse.ArgumentParser(description='Define Inputs for pruning model')
parser.add_argument('-i', action="store", dest="Iteration")
results = parser.parse_args()
try:
    iteration = int(results.Iteration)
    print('Current Iteration : ', iteration)
except:
    raise Exception('Arguement not supplied')

PATH_ENCODER = 'encoder_aggregated_' + str(iteration)
PATH_REGRESSOR = 'regressor_aggregated_' + str(iteration)
PATH_DOMAIN = 'domain_aggregated_' + str(iteration)

########################################################################################################################
weights = {'a': 1, 'b': 1, 'c': 1, 'd':1 }
total = 4

print(weights)
sites = ['Trinity', 'NYU', 'UCLA', 'Yale']
site_dict = {'Trinity': 'a', 'NYU':'b', 'UCLA':'c', 'Yale':'d'}

encoder_pths = []
regressor_pths = []
domain_pths = []

for i, site in enumerate(sites):
    encoder_pths.append('encoder_' + site + '_' + str(iteration))
    regressor_pths.append('regressor_' + site + '_' + str(iteration))
    domain_pths.append('domain_' + site + '_' + str(iteration))

encoder_pths = np.array(encoder_pths)
regressor_pths = np.array(regressor_pths)
domain_pths = np.array(domain_pths)

update_state_encoder = OrderedDict()
update_state_regressor = OrderedDict()
update_state_domain = OrderedDict()

encoder = Encoder()
regressor = Regressor()
domain = DomainPredictor(4)

if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()
    domain = domain.cuda()

for i, site in enumerate(sites):
    encoder.load_state_dict(torch.load(encoder_pths[i]))
    regressor.load_state_dict(torch.load(regressor_pths[i]))
    domain.load_state_dict(torch.load(domain_pths[i]))

    local_state_encoder = encoder.state_dict()
    local_state_regressor = regressor.state_dict()
    local_state_domain = domain.state_dict()
    s = site_dict[site]
    for key in encoder.state_dict().keys():
        if i == 0:
            update_state_encoder[key] = local_state_encoder[key] * (weights[s]/total)
        else:
            update_state_encoder[key] += local_state_encoder[key] * (weights[s] / total)
    for key in regressor.state_dict().keys():
        if i == 0:
            update_state_regressor[key] = local_state_regressor[key] * (weights[s]/total)
        else:
            update_state_regressor[key] += local_state_regressor[key] * (weights[s]/total)
    for key in domain.state_dict().keys():
        if i == 0:
            update_state_domain[key] = local_state_domain[key] * (weights[s]/total)
        else:
            update_state_domain[key] += local_state_domain[key] * (weights[s]/total)

encoder.load_state_dict(update_state_encoder)
regressor.load_state_dict(update_state_regressor)
domain.load_state_dict(update_state_domain)

torch.save(encoder.state_dict(), PATH_ENCODER)
torch.save(regressor.state_dict(), PATH_REGRESSOR)
torch.save(domain.state_dict(), PATH_DOMAIN)

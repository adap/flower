"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision.models import resnet
from torchvision import models
from fedsmoo.minimizers import *

# class GroupNorm_(nn.Module):
#     def __init__(self, num_channels):
#         super(GroupNorm_, self).__init__()
#         self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
#                                  eps=1e-5, affine=True)
    
#     def forward(self, x):
#         x = self.norm(x)
#         return x


# class ResNet18GN(nn.Module):
#     """
#     ResNet18 with batchnorm layers replaced by groupnorm
#     """

#     def __init__(self, num_classes: int) -> None:
#         # set the batchnorm layers to group norm layers in model 
#         super().__init__()
#         self.resnet18 = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=GroupNorm_)


#     def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         ResNet18GN forward pass

#         Parameters
#         --------------
#         input_tensor : torch.Tensor input images fed through the network

#         Returns
#         --------------
#         torch.Tensor
#             model output
#         """

#         x = self.resnet18(input_tensor)
#         return x

class ResNet18GN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18GN, self).__init__()
        resnet18 = models.resnet18()
        resnet18.fc = nn.Linear(512, 100)
        
        # Change BN to GN 
        resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
        
        self.model = resnet18
    
    def forward(self, input_tensor):            
        x = self.model(input_tensor)
        return x
    


# ----------------- Train FedSMOO ------------------------------------------
def trainFedSMOO(net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    alpha_coef_adpt: float,
    init_mdl_param: torch.tensor, # 1D
    hist_params_diff: torch.tensor,   # lambda
    hist_sam_diffs_list: List[torch.tensor],  # mu
    gs_diff_list: List[torch.tensor], 
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    sch_step: int,
    sch_gamma: float,
    sam_lr: float,    # r in the algo
    ):
    """
    Call the train function to train the network on the train dataset.

    Parameters
    -----------------------------
    net: nn.Module
        pytorch resent model loaded with server model weights

    trainloader: Dataloader
        train dataloader for client training

    device: torch.device
        model loading device, cpu/cuda

    alpha_coef_adpt: float
        (check FedDyn loss weights*grad, also used in SGD weight decay)
    
    init_mdl_param: torch.tensor (1D)
        server model initial params as a torch 1d tensor
    
    hist_params_diff: torch.tensor (1D)
        client hist params diff as torch 1d tensor 
        [lambda dual for weights, 
            stores difference between weights before, after training]

    hist_sam_diffs_list: List[torch.tensor]
        mu dual variable for client as list of torch weight matrices
    
    gs_diff_list: List[torch.tensor]
        s global perturbation shared by the server as list of torch weight matrcies

    local_epochs: int
        number of local training iterations over the train dataset

    learning_rate: float
        local model learning rate for the current round
        # initial_learning_rate * (lr_decay_per_round ** i)
    
    weight_decay: float
        SGD weight decay parameter
    
    sch_step: int
        step_size parameter StepLR (how many epochs until lr is updated) 
    
    sch_gamma: float
        StepLR scheduler step multiplication factor
    
    samlr: float
        SAM perturbation learning rate

    Returns
    net: 
        torch model
    hist_sam_diffs_list:
        list of torch tensors
    """

    # get model with updated weights, hist_sam_diffs_list (list of torch tensors)
    net, hist_sam_diffs_list = _train_model_GS(device, net, alpha_coef_adpt, init_mdl_param, 
                                                 hist_params_diff, hist_sam_diffs_list, gs_diff_list,
                                                 trainloader, learning_rate, local_epochs, 
                                                 weight_decay, sch_step, sch_gamma, sam_lr)
    
    return net, hist_sam_diffs_list

def _train_model_GS(device, model, alpha_coef, init_mdl_param, 
                    hist_params_diff, hist_sam_diff_list, gs_diff_curr_list,
                    trainloader, learning_rate, epoch,
                    weight_decay, sch_step, sch_gamma, samlr, max_norm=10):

    init_mdl_param = init_mdl_param.to(device)
    hist_params_diff = hist_params_diff.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=sch_step, gamma=sch_gamma)
    
    model.train()
    model = model.to(device)
    optimizer = GSAM(device, model.parameters(), base_optimizer, rho=samlr, beta=1.0, gamma=1.0, adaptive=False,
                     nograd_cutoff=0.05)
    
    model.train()

    for e in range(epoch):
        # Training
        for batch_x, batch_y in trainloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.reshape(-1).long()
            
            def defined_backward(loss):
                loss.backward()
            paras = [batch_x, batch_y, loss_fn, model, defined_backward]
            optimizer.paras = paras
            hist_sam_diff_list = optimizer.step(hist_sam_diff_list, gs_diff_curr_list)
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-init_mdl_param + hist_params_diff))
            
            loss = loss_algo

            ###
            # base_optimizer.zero_grad() # why has this been commented?
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            base_optimizer.step()
            
        model.train()
        scheduler.step()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, hist_sam_diff_list

# ------------------ Train FedAvg ------------------------------------------

def trainFedAvg(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    net.train()
    for _ in range(epochs):
        net = _training_fedavg_epoch(net, trainloader, device, criterion, optimizer)

def _training_fedavg_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    max_norm: float = 10) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=max_norm)
        optimizer.step()
    return net


# ------------------ Train FedDyn ------------------------------------------

# def trainFedDyn():
#     pass 

# def _train_feddyn_epoch():
#     pass

# ------------------- FedSMOO Utils ----------------------------------------------

def set_H_param_list(model, m, device=None):
    """
    Parameters
    ---------------------------------
    model: ResNet18GN
        torch model for boilerplate tensor shapes
    
    m: np.array
        1d array of flattened model weights
    
    Returns 
    ---------------------------------
    H_element_list: List[torch.tensor]
        list of model torch tensor weight matrices

    """
    H_element_list = []
    idx = 0
    for name, param in model.named_parameters():
        length = len(param.data.reshape(-1))
        H_element_list.append(torch.tensor(m[idx:idx + length].reshape(param.data.shape), dtype=torch.float32, device=device, requires_grad=False))
        idx += length
    return H_element_list

def get_H_param_array(m):
    """
    given list of params get flattened numpy array (n,)
    """
    local_array = None
    for param in m:
        if not isinstance(local_array, torch.Tensor):
            local_array = param.reshape(-1)
        else:
            local_array = torch.cat((local_array, param.reshape(-1)), 0)

    local_array = local_array.cpu().numpy()
    return local_array

def get_mdl_params(model_list, n_par=None):
    """
    Returns a list of 1d numpy arrays of parameters given torch model
    """
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

# --------------------------------------------------------------------------------

def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    
    """
    Evaluate the client networks

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data
    """

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    if len(testloader.dataset) == 0:
        return 0.0, 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


if __name__=="__main__":
    model = ResNet18GN(10)
    n = sum(p.numel() for p in model.parameters())
    print(n)

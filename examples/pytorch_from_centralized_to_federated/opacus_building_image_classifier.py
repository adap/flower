#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Building-an-Image-Classifier-with-Differential-Privacy" data-toc-modified-id="Building-an-Image-Classifier-with-Differential-Privacy-1">Building an Image Classifier with Differential Privacy</a></span><ul class="toc-item"><li><span><a href="#Overview" data-toc-modified-id="Overview-1.1">Overview</a></span></li><li><span><a href="#Hyper-parameters" data-toc-modified-id="Hyper-parameters-1.2">Hyper-parameters</a></span></li><li><span><a href="#Data" data-toc-modified-id="Data-1.3">Data</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-1.4">Model</a></span></li><li><span><a href="#Prepare-for-Training" data-toc-modified-id="Prepare-for-Training-1.5">Prepare for Training</a></span></li><li><span><a href="#Train-the-network" data-toc-modified-id="Train-the-network-1.6">Train the network</a></span></li><li><span><a href="#Test-the-network-on-test-data" data-toc-modified-id="Test-the-network-on-test-data-1.7">Test the network on test data</a></span></li><li><span><a href="#Tips-and-Tricks" data-toc-modified-id="Tips-and-Tricks-1.8">Tips and Tricks</a></span></li><li><span><a href="#Private-Model-vs-Non-Private-Model-Performance" data-toc-modified-id="Private-Model-vs-Non-Private-Model-Performance-1.9">Private Model vs Non-Private Model Performance</a></span></li></ul></li></ul></div>

# # Building an Image Classifier with Differential Privacy

# ## Overview
#
# In this tutorial we will learn to do the following:
#   1. Learn about privacy specific hyper-parameters related to DP-SGD
#   2. Learn about ModelInspector, incompatible layers, and use model rewriting utility.
#   3. Train a differentially private ResNet18 for image classification.

# ## Hyper-parameters

# To train a model with Opacus there are three privacy-specific hyper-parameters that must be tuned for better performance:
#
# * Max Grad Norm: The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.
# * Noise Multiplier: The amount of noise sampled and added to the average of the gradients in a batch.
# * Delta: The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. In this tutorial, it is set to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.
#
# We use the hyper-parameter values below to obtain results in the last section:

# In[1]:
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 10

LR = 1e-3
NUM_WORKERS = 4


# There's another constraint we should be mindful of&mdash;memory. To balance peak memory requirement, which is proportional to `batch_size^2`, and training performance, we use virtual batches. With virtual batches we can separate physical steps (gradient computation) and logical steps (noise addition and parameter updates): use larger batches for training, while keeping memory footprint low. Below we will specify two constants:

# In[2]:


BATCH_SIZE = 32
VIRTUAL_BATCH_SIZE = 64
assert VIRTUAL_BATCH_SIZE % BATCH_SIZE == 0 # VIRTUAL_BATCH_SIZE should be divisible by BATCH_SIZE
N_ACCUMULATION_STEPS = int(VIRTUAL_BATCH_SIZE / BATCH_SIZE)


# ## Data

# Now, let's load the CIFAR10 dataset. We don't use data augmentation here because, in our experiments, we found that data augmentation lowers utility when training with DP.

# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms

from cifar import load_data_set

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budget.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])


# Using torchvision datasets, we can load CIFAR10 and transform the PILImage images to Tensors of normalized range [-1, 1]

# In[4]:

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

train_dataset, test_dataset = load_data_set()

SAMPLE_RATE = BATCH_SIZE / len(train_dataset)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=NUM_WORKERS,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(train_dataset),
        sample_rate=SAMPLE_RATE,
    ),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ## Model

# In[5]:
from torchvision import models

from cifar import Net

#model = models.resnet18(num_classes=10)
model = Net()

# Now, let’s check if the model is compatible with Opacus. Opacus does not support all type of Pytorch layers. To check if your model is compatible with the privacy engine, we have provided a util class to validate your model.

# If you run these commands, you will get the following error:

# In[6]:


from opacus.dp_model_inspector import DPModelInspector

#inspector = DPModelInspector()
#inspector.validate(model)


# Let us modify the model to work with Opacus. From the output above, you can see that the BatchNorm layers are not supported because they compute the mean and variance across the batch, creating a dependency between samples in a batch, a privacy violation. One way to modify our model is to replace all the BatchNorm layers with [GroupNorm](https://arxiv.org/pdf/1803.08494.pdf) using the `convert_batchnorm_modules` util function.

# In[7]:


from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification

model = module_modification.convert_batchnorm_modules(model)
inspector = DPModelInspector()
print(f"Is the model valid? {inspector.validate(model)}")


# For maximal speed, we can check if CUDA is available and supported by the PyTorch installation. If GPU is available, set the `device` variable to your CUDA-compatible device. We can then transfer the neural network onto that device.

# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


# We then define our optimizer and loss function. Opacus’ privacy engine can attach to any (first-order) optimizer.  You can use your favorite&mdash;Adam, Adagrad, RMSprop&mdash;as long as it has an implementation derived from [torch.optim.Optimizer](https://pytorch.org/docs/stable/optim.html). In this tutorial, we're going to use [RMSprop](https://pytorch.org/docs/stable/optim.html).

# In[9]:


import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)


# ## Prepare for Training

# We will define a util function to calculate accuracy

# In[10]:


def accuracy(preds, labels):
    return (preds == labels).mean()


# We now attach the privacy engine initialized with the privacy hyperparameters defined earlier.

# In[11]:


from opacus import PrivacyEngine

privacy_engine = PrivacyEngine(
    model,
    sample_rate=SAMPLE_RATE * N_ACCUMULATION_STEPS,
    epochs = EPOCHS,
    target_epsilon = EPSILON,
    target_delta = DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

privacy_engine.attach(optimizer)
print(f"Using sigma={privacy_engine.noise_multiplier} and C={MAX_GRAD_NORM}")


# We will then define our train function. This function will train the model for one epoch.

# In[12]:


import numpy as np

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)

        loss.backward()

        # take a real optimizer step after N_VIRTUAL_STEP steps t
        if ((i + 1) % N_ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()
        else:
            optimizer.virtual_step() # take a virtual step

        if i % 200 == 0:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(DELTA)
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {DELTA})"
            )


# Next, we will define our test function to validate our model on our test dataset.

# In[13]:


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)


# ## Train the network

# In[ ]:
from tqdm import tqdm

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    train(model, train_loader, optimizer, epoch + 1, device)


# ## Test the network on test data

# In[ ]:


top1_acc = test(model, test_loader, device)


# ## Tips and Tricks

# 1. Generally speaking, differentially private training is enough of a regularizer by itself. Adding any more regularization (such as dropouts or data augmentation) is unnecessary and typically hurts performance.
# 2. Tuning MAX_GRAD_NORM is very important. Start with a low noise multiplier like .1, this should give comparable performance to a non-private model. Then do a grid search for the optimal MAX_GRAD_NORM value. The grid can be in the range [.1, 10].
# 3. You can play around with the level of privacy, EPSILON.  Smaller EPSILON means more privacy, more noise -- and hence lower accuracy.  Reducing EPSILON to 5.0 reduces the Top 1 Accuracy to around 53%.  One useful technique is to pre-train a model on public (non-private) data, before completing the training on the private training data.  See the workbook at [bit.ly/opacus-dev-day](https://bit.ly/opacus-dev-day) for an example.
#

# ## Private Model vs Non-Private Model Performance

# Now let us compare how our private model compares with the non-private ResNet18.
#
# We trained a non-private ResNet18 model for 20 epochs using the same hyper-parameters as above and with BatchNorm replaced with GroupNorm. The results of that training and the training that is discussed in this tutorial are summarized in the table below:

# | Model          | Top 1 Accuracy (%) |  ϵ |
# |----------------|--------------------|---|
# | ResNet         | 76                 | ∞ |
# | Private ResNet |         56.61         |  53.54  |

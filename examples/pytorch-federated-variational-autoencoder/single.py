from utils_mnist import train, test, test2, train_and_eval, Net, load_data_mnist, VAE
import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

IDENTIFIER = "single_app_cpu"
if not os.path.exists(IDENTIFIER):
    os.makedirs(IDENTIFIER)
epochs = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = load_data_mnist(normalise=False)
net = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2).to(DEVICE)
model = net
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
config = {"identifier": IDENTIFIER}
# Train
train_and_eval(
    model,
    train_loader,
    optimizer,
    config,
    epochs=epochs,
    device=DEVICE,
    num_classes=None,
    testloader=val_loader,
)


with torch.no_grad():
    z = torch.randn(64, 2).cuda()
    sample = net.decoder(z).cuda()

    save_image(sample.view(64, 1, 28, 28), "./samples/sample_" + ".png")

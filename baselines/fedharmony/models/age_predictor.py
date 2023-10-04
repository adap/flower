# Nicola Dinsdale 2022
# Model for unlearning domain
########################################################################################################################
# Import dependencies
import torch.nn as nn
########################################################################################################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature = nn.Sequential()      # Define the feature extractor
        self.feature.add_module('f_conv1_1', nn.Conv3d(1, 4, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_1_1', nn.ReLU(True))
        self.feature.add_module('f_bn1_1', nn.BatchNorm3d(4))
        self.feature.add_module('f_conv1_2', nn.Conv3d(4, 4, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_1_2', nn.ReLU(True))
        self.feature.add_module('f_bn1_2', nn.BatchNorm3d(4))
        self.feature.add_module('f_pool1', nn.MaxPool3d(2))

        self.feature.add_module('f_conv2_1', nn.Conv3d(4, 8, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_2_1', nn.ReLU(True))
        self.feature.add_module('f_bn2_1', nn.BatchNorm3d(8))
        self.feature.add_module('f_conv2_2', nn.Conv3d(8, 8, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_2_2', nn.ReLU(True))
        self.feature.add_module('f_bn2_2', nn.BatchNorm3d(8))
        self.feature.add_module('f_pool2', nn.MaxPool3d(2))

        self.feature.add_module('f_conv3_1', nn.Conv3d(8, 8, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_3_1', nn.ReLU(True))
        self.feature.add_module('f_bn3_1', nn.BatchNorm3d(8))
        self.feature.add_module('f_conv3_2', nn.Conv3d(8, 8, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_3_2', nn.ReLU(True))
        self.feature.add_module('f_bn3_2', nn.BatchNorm3d(8))
        self.feature.add_module('f_pool3', nn.MaxPool3d(2))

        self.feature.add_module('f_conv4_1', nn.Conv3d(8, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_4_1', nn.ReLU(True))
        self.feature.add_module('f_bn4_1', nn.BatchNorm3d(16))
        self.feature.add_module('f_conv4_2', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_4_2', nn.ReLU(True))
        self.feature.add_module('f_bn4_2', nn.BatchNorm3d(16))
        self.feature.add_module('f_conv4_3', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_4_3', nn.ReLU(True))
        self.feature.add_module('f_bn4_3', nn.BatchNorm3d(16))
        self.feature.add_module('f_pool4', nn.MaxPool3d(2))

        self.feature.add_module('f_conv5_1', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_5_1', nn.ReLU(True))
        self.feature.add_module('f_bn5_1', nn.BatchNorm3d(16))
        self.feature.add_module('f_conv5_2', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_5_2', nn.ReLU(True))
        self.feature.add_module('f_bn5_2', nn.BatchNorm3d(16))
        self.feature.add_module('f_conv5_3', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_5_3', nn.ReLU(True))
        self.feature.add_module('f_bn5_3', nn.BatchNorm3d(16))
        self.feature.add_module('f_pool5', nn.MaxPool3d(2))

        self.feature.add_module('f_conv6_1', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_6_1', nn.ReLU(True))
        self.feature.add_module('f_bn6_1', nn.BatchNorm3d(16))
        self.feature.add_module('f_conv6_2', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_6_2', nn.ReLU(True))
        self.feature.add_module('f_bn6_2', nn.BatchNorm3d(16))
        self.feature.add_module('f_conv6_3', nn.Conv3d(16, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_relu_6_3', nn.ReLU(True))
        self.feature.add_module('f_bn6_3', nn.BatchNorm3d(16))
        self.feature.add_module('f_features', nn.MaxPool3d(2))

        self.embeddings = nn.Sequential()
        self.embeddings.add_module('r_fc1', nn.Linear(192, 64))
        self.embeddings.add_module('r_relu1', nn.ReLU(True))

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(-1, 192)
        feature_embedding = self.embeddings(feature)
        return feature_embedding

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc2', nn.Linear(64, 16))
        self.regressor.add_module('r_relu2', nn.ReLU(True))
        self.regressor.add_module('r_pred', nn.Linear(16, 1))

    def forward(self, x):
        regression = self.regressor(x)
        return regression

class DomainPredictor(nn.Module):
    def __init__(self, nodes=2):
        super(DomainPredictor, self).__init__()
        self.nodes = nodes
        self.domain = nn.Sequential()
        self.domain.add_module('d_fc2', nn.Linear(64, 32))
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout3d(p=0.2))
        self.domain.add_module('d_fc3', nn.Linear(32, nodes))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        domain_pred = self.domain(x)
        return domain_pred




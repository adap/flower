import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d


class STDDAGMM(nn.Module):    
    def __init__(self, encoder_dims, n_gmm_components=3, lambda_1=0.1, lambda_2=0.005, dropout_rate=0.5):
        super(STDDAGMM, self).__init__()
            
        # Build encoder with batch normalization
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:  # Don't add BN and activation after last layer
                self.encoder.append(BatchNorm1d(encoder_dims[i + 1]))
                self.encoder.append(nn.ReLU())  # Using ReLU instead of Tanh for better gradient flow
        
        # Build decoder (mirror of encoder)
        decoder_dims = list(reversed(encoder_dims))
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                self.decoder.append(BatchNorm1d(decoder_dims[i + 1]))
                self.decoder.append(nn.ReLU())
        
        # Estimation network for GMM parameters
        # Now accepts 5 features: STD + cosine + euclidean + relative euclidean + reconstruction error variance
        latent_dim = encoder_dims[-1]
        estimator_input_dim = latent_dim + 5
        
        # More sophisticated estimation network
        self.estimator = nn.Sequential(
            nn.Linear(estimator_input_dim, 128),
            BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, n_gmm_components),
            nn.Softmax(dim=1)
        )
        
        self.n_gmm_components = n_gmm_components
        self.latent_dim = latent_dim
        self.criterion = STDDAGMMLoss(lambda_1, lambda_2)
    
    def forward(self, x):
        compressed = self._encode(x)
        reconstruction = self._decode(compressed)
        features = self._compute_features(x, reconstruction, compressed)
        gamma = self._estimate(features)
        error = F.mse_loss(x, reconstruction, reduction='none').mean(dim=1)
        
        return gamma, features, error
    
    def _encode(self, x):
        h = x
        for layer in self.encoder:
            h = layer(h)
        return h
    
    def _decode(self, compressed):
        h = compressed
        for layer in self.decoder:
            h = layer(h)
        return h
    
    def _estimate(self, features):
        return self.estimator(features)
    
    def _compute_features(self, input, reconstruction, compressed):
        """Compute enhanced features for anomaly detection"""
        batch_size = input.size(0)
        
        std_input = torch.std(input, dim=1, keepdim=True)
        cosine_sim = F.cosine_similarity(input, reconstruction, dim=1).unsqueeze(1)
        euclidean_dist = torch.norm(input - reconstruction, p=2, dim=1, keepdim=True)
        input_norm = torch.norm(input, p=2, dim=1, keepdim=True)
        relative_euclidean = euclidean_dist / (input_norm + 1e-8)
        error_variance = torch.var(input - reconstruction, dim=1, keepdim=True)
        
        # Normalize features to similar scales
        features = torch.cat([
            self._normalize_feature(std_input),
            cosine_sim,  # Already in [-1, 1]
            self._normalize_feature(euclidean_dist),
            relative_euclidean,  # Already normalized
            self._normalize_feature(error_variance)
        ], dim=1)
        
        combined_features = torch.cat([compressed, features], dim=1)
        
        return combined_features
    
    def _normalize_feature(self, feature):
        """Normalize feature to approximately [0, 1] range"""
        min_val = feature.min()
        max_val = feature.max()
        if max_val - min_val > 1e-8:
            return (feature - min_val) / (max_val - min_val + 1e-8)
        return feature


class STDDAGMMLoss(nn.Module):
    """Enhanced loss function with numerical stability improvements"""
    
    def __init__(self, lambda_1, lambda_2):
        super(STDDAGMMLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.eps =1e-3
        
    def forward(self, gamma, features, reconstruction_error):
        mixture_probs, means, covariances = self._get_gmm_parameters(gamma, features)
        reconstruction_loss = torch.mean(reconstruction_error)
        energy_loss = torch.mean(self._compute_energy(features, mixture_probs, means, covariances))
        singularity_loss = self._compute_singularity_loss(covariances)

        total_loss = (reconstruction_loss + 
                     self.lambda_1 * energy_loss + 
                     self.lambda_2 * singularity_loss)
        
        return total_loss
    
    def _get_gmm_parameters(self, gamma, features):
        """Compute GMM parameters from membership probabilities"""
        batch_size, n_components = gamma.size()
        feature_dim = features.size(1)
        
        # Mixture probabilities (phi)
        phi = torch.mean(gamma, dim=0)
        
        # Means (mu)
        mu = torch.sum(gamma.unsqueeze(2) * features.unsqueeze(1), dim=0)
        mu = mu / (torch.sum(gamma, dim=0).unsqueeze(1) + 1e-8)
        
        # Covariances (sigma)
        diff = features.unsqueeze(1) - mu.unsqueeze(0)
        weighted_diff = gamma.unsqueeze(2) * diff
        sigma = torch.bmm(weighted_diff.permute(1, 2, 0), diff.permute(1, 0, 2))
        sigma = sigma / (torch.sum(gamma, dim=0).unsqueeze(1).unsqueeze(2) + 1e-8)
        
        # Add small diagonal term for numerical stability
        eye = torch.eye(feature_dim, device=sigma.device).unsqueeze(0)
        sigma = sigma + 1e-6 * eye
        
        return phi, mu, sigma
    
    def _compute_energy(self, features, phi, mu, sigma):
        """Compute sample energy under the GMM with numerical stability"""
        batch_size = features.size(0)
        n_components = phi.size(0)
        feature_dim = features.size(1)
        
        energies = []
        for i in range(n_components):
            diff = features - mu[i].unsqueeze(0)
            
            try:
                L = torch.linalg.cholesky(sigma[i])
                inv_sigma = torch.cholesky_inverse(L)
            except:
                try:
                    inv_sigma = torch.linalg.pinv(sigma[i])
                except:
                    diag = torch.diagonal(sigma[i])
                    inv_sigma = torch.diag(1.0 / (diag + self.eps))
            
            mahal_dist = torch.sum(diff @ inv_sigma * diff, dim=1)
            
            try:
                log_det = torch.logdet(sigma[i])
                if torch.isnan(log_det) or torch.isinf(log_det):
                    log_det = torch.sum(torch.log(torch.diagonal(sigma[i]) + self.eps))
            except:
                log_det = torch.sum(torch.log(torch.diagonal(sigma[i]) + self.eps))
            
            log_2pi = feature_dim * torch.log(torch.tensor(2 * 3.14159, device=features.device))
            log_prob = -0.5 * (mahal_dist + log_det + log_2pi)
            
            weighted_log_prob = torch.log(phi[i] + 1e-10) + log_prob
            energies.append(weighted_log_prob)
        
        energy_stack = torch.stack(energies, dim=0)
        max_energy = torch.max(energy_stack, dim=0)[0]
        
        sum_exp = torch.sum(torch.exp(energy_stack - max_energy.unsqueeze(0)), dim=0)
        total_energy = -(max_energy + torch.log(sum_exp + 1e-10))
        
        total_energy = torch.where(torch.isfinite(total_energy), total_energy, torch.zeros_like(total_energy))
        
        return total_energy
    
    def _compute_singularity_loss(self, sigma):
        """Prevent covariance matrices from becoming singular"""
        diag_elements = torch.diagonal(sigma, dim1=-2, dim2=-1)
        return torch.sum(1.0 / (diag_elements + 1e-8))


def train_std_dagmm(net, data_loader, device, epochs=100, learning_rate=1e-3):
    """Training function for Improved STD-DAGMM"""
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    net.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            gamma, features, error = net(data)
            loss = net.criterion(gamma, features, error)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


class DAGMM(nn.Module):
    def __init__(self, n_encoder_layers, n_gmm_layers, lambda_1=0.1, lambda_2=0.005):
        super(DAGMM, self).__init__()

        self.encoder = self._build_module_list(n_encoder_layers[:-1], nn.Tanh())
        self.encoder.append(nn.Linear(n_encoder_layers[-2], n_encoder_layers[-1]))

        self.decoder = self._build_module_list(list(reversed(n_encoder_layers[1:])), nn.Tanh())
        self.decoder.append(nn.Linear(n_encoder_layers[1], n_encoder_layers[0]))

        self.estimator = nn.ModuleList()
        self.estimator.append(nn.Linear(n_encoder_layers[-1] + 3, n_gmm_layers[0]))
        self.estimator.extend(self._build_module_list(n_gmm_layers[:-1], nn.Tanh()))
        self.estimator.append(nn.Dropout(0.5))
        self.estimator.append(nn.Linear(n_gmm_layers[-2], n_gmm_layers[-1]))
        self.estimator.append(nn.Softmax(dim=1))

        self.criterion = DAGMMLoss(lambda_1, lambda_2)

    def forward(self, x):
        compressed = self._encode(x)
        reconstruction = self._decode(compressed)
        combined_features = torch.cat((compressed, self._get_features(x, reconstruction)), dim=1)
        error = torch.cdist(x.permute(1, 0).unsqueeze(1), reconstruction.permute(1, 0).unsqueeze(1), p=2).squeeze()
        return self._estimate(combined_features), combined_features, error

    def _encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def _decode(self, compressed):
        for layer in self.decoder:
            compressed = layer(compressed)
        return compressed

    def _estimate(self, combined_features):
        for layer in self.estimator:
            combined_features = layer(combined_features)
        return combined_features

    @staticmethod
    def _get_features(input, reconstruction):
        euclidean_distance = torch.cdist(input.unsqueeze(1), reconstruction.unsqueeze(1), p=2).squeeze()
        return torch.stack((torch.std(input, dim=1), torch.cosine_similarity(input, reconstruction), euclidean_distance), dim=1)

    @staticmethod
    def _build_module_list(layers, activation):
        modules = []
        for in_dim, out_dim in zip(layers, layers[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(activation)
        return nn.ModuleList(modules)


class DAGMMLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2):
        super(DAGMMLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, membership_estimations, combined_features, error):
        mixture_probabilities, means, covariances = get_gmm_parameters(membership_estimations, combined_features)
        error_component = torch.mean(error)
        energy_component = torch.mean(
            torch.stack([get_energy(f, mixture_probabilities, means, covariances) for f in combined_features]), dim=0
        )
        diagonal_component = torch.sum(torch.reciprocal(torch.diagonal(covariances, dim1=-2, dim2=-1)))
        return error_component + self.lambda_1 * energy_component + self.lambda_2 * diagonal_component


def train_dagmm(net, data_loader, device, epochs=100):
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        for batch in data_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            membership_estimations, combined_features, error = net(data)
            loss = net.criterion(membership_estimations, combined_features, error)
            loss.backward()
            optimizer.step()


def get_energy(sample, mixture_probabilities, means, covariances):
    diffs = sample - means
    term1 = torch.log(mixture_probabilities)
    term2 = (-0.5 * diffs.unsqueeze(1) @ torch.pinverse(covariances) @ diffs.unsqueeze(2)).squeeze()
    term3 = -0.5 * torch.logdet(2 * torch.pi * covariances)
    energy = -torch.logsumexp(term1 + term2 + term3, dim=0)
    return energy


def get_gmm_parameters(membership_estimations, combined_features):
    mixture_probabilities = membership_estimations.mean(dim=0)
    weighted_features = membership_estimations.T @ combined_features
    means = weighted_features / membership_estimations.sum(dim=0, keepdim=True).T
    diffs = combined_features.unsqueeze(0) - means.unsqueeze(1)
    covariances = torch.sum(
        membership_estimations.T.unsqueeze(-1).unsqueeze(-1) * (diffs.unsqueeze(-1) @ diffs.unsqueeze(-2)), dim=1
    ) / membership_estimations.sum(dim=0, keepdim=True).T.unsqueeze(-1)
    covariances.diagonal(dim1=-1, dim2=-2).add_(1e-6)
    return mixture_probabilities, means, covariances

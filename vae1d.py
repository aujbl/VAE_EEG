import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchsummary import summary


class VAE(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, **kargs):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32]

        # build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_features=h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # build decoder
        modules = []
        in_channels = latent_dim
        hidden_dims.reverse()

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_features=h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.in_channels),
            nn.LeakyReLU(),
        )

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss

        return {"loss": loss, "recons_loss": recons, "kld_loss": kld_loss}

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        # to device?
        return self.decode(z)

    def generate(self, x, **kargs):
        return self.forward(x)[0]


if __name__ == '__main__':
    vae = VAE(in_channels=68, latent_dim=32, hidden_dims=[64, 32])
    summary(vae, (1000, 68))

















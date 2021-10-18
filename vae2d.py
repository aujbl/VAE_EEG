import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchsummary import summary


class VAE2d(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, **kargs):
        super(VAE2d, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [4, 8, 16]

        # build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    # consider kernel_size == (1, 7) or sth else
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules,
                                     nn.Flatten())
        # flatten size input
        fla_size = hidden_dims[-1] * 4
        self.fc_mu = nn.Linear(fla_size, latent_dim)
        self.fc_var = nn.Linear(fla_size, latent_dim)

        # build decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, fla_size)
        hidden_dims.reverse()

        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, self.in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.Tanh(),
        )
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh())

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
    # summary(vae, (1000, 68))

















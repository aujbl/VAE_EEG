import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F


class MyVAE(nn.Module):

    def __init__(self, in_channels, latent_dim, channels, **kwargs):
        super(MyVAE, self).__init__()

        self.latent_dim = latent_dim

        self.channels = [1, 32, 32, 64, 64, 128, 128]
        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=channels[1], out_channels=channels[2],
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=channels[2], out_channels=channels[3],
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=channels[3], out_channels=channels[4],
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[4]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=channels[4], out_channels=channels[5],
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[5]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=channels[5], out_channels=channels[6],
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[6]),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )
        # self.flatten_size = self.encoder(torch.randn(1, 1, 68, 1000)).shape
        self.flatten_size = 7936
        # self.interLinear Layer
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        # self.decoder_input.view(-1, channels[-1], 2, 31)

        channels.reverse()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=channels[1],
                               kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[2],
                               kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[3],
                               kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[4],
                               kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[4]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[5],
                               kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[5]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(in_channels=channels[5], out_channels=channels[6],
                               kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(channels[6]),
            nn.LeakyReLU(),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

        )

        # self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.channels[-1],
                                               out_channels=1,  # hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               # padding='same',
                                               # output_padding=1
                                               ),
                            nn.BatchNorm2d(1),
                            nn.LeakyReLU(),
                            # nn.Conv2d(hidden_dims[-1],
                            #           out_channels=1,
                            #           kernel_size=3,
                            #           # padding='same'
                            #           ),
                            # nn.Tanh()
                            )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.channels[-1], 2, 31)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the mini batch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == '__main__':
    my_vae = MyVAE(in_channels=1, latent_dim=32, channels=[1, 32, 32, 64, 64, 128, 128])
    # summary(MyVAE)
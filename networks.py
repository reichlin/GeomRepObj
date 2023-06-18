import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class ResNet18(nn.Module):

    def __init__(self, N, channels, out_dims):
        super().__init__()

        self.channels = channels

        if N == 100:
            self.dim_f = 512 * 4 * 4

        if N == 224:
            self.dim_f = 512 * 7 * 7

        if self.channels != 3:

            self.first_block = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=0),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU())

        self.conv_stride = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        resnet = models.resnet18(pretrained=False)
        self.base_model = resnet
        self.fc1 = nn.Linear(self.dim_f, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dims)

    def forward(self, x):

        if self.channels != 3:
            x = self.first_block(x)
        else:
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = F.relu(self.conv_stride(x))

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = x.view(-1, self.dim_f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ConvNet(nn.Module):

    def __init__(self, N, channels, out_dims):
        super().__init__()

        self.channels = channels

        if N == 100:
            self.dim_f = 32 * 9 * 9

        self.net = nn.Sequential(nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=0),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=0),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(32 * 9 * 9, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, out_dims))

    def forward(self, x):
        return self.net(x)


class EasyConv(nn.Module):

    def __init__(self, N, channels, out_dims):
        super().__init__()

        self.channels = channels

        self.dim_f = 128 * 6 * 6

        self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(self.dim_f, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, out_dims))

    def forward(self, x):
        return self.net(x)


class Base_Decoder(nn.Module):

    def __init__(self, N, in_dim, channels):
        super().__init__()

        if N == 100:
            self.dim_f = [32, 11, 11]
            self.fc = nn.Sequential(nn.Linear(in_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.dim_f[0] * self.dim_f[1] * self.dim_f[2]))

            self.decoder = nn.Sequential(nn.ConvTranspose2d(self.dim_f[0], 64, 3, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, 64, 4, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, 64, 4, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(64, channels, 3, stride=1, padding=0))
        else:
            print("Not implemented yet")
            exit()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_f[0], self.dim_f[1], self.dim_f[2])
        return self.decoder(x)

class VAE(nn.Module):

    def __init__(self, N, channels, z_dim, h_features, rl=False):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.h_features = h_features
        self.rl = rl

        self.encoder = ResNet18(N, channels, z_dim+2*h_features)
        self.decoder = Base_Decoder(N, z_dim+h_features, channels)

    def forward(self, o):

        x = self.encoder(o)
        grip = x[:, :self.z_dim]
        mu = torch.sigmoid(x[:, self.z_dim:self.z_dim+self.h_features])
        logvar = x[:, self.h_features+self.z_dim:]

        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu

        dec_in = torch.cat([grip, z], -1)
        o_hat = self.decoder(dec_in)

        if self.rl:
            return torch.cat([grip, mu], -1)

        return grip, mu, logvar, o_hat

    def get_vae_loss(self, o, o_hat, mu, logvar):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        mse_loss = torch.mean(torch.sum((o - o_hat)**2, (1, 2, 3)))
        return mse_loss, kl_loss


class phi(nn.Module):

    def __init__(self, N, channels, a_dim, n_objs, learn_var, rotations, eigen_vals, device, rl=False):
        super().__init__()

        self.a_dim = a_dim
        self.n_objs = n_objs
        self.dim_touch = 8

        self.rotations = rotations

        self.device = device

        self.rl = rl

        self.encoder_z = ResNet18(N, channels, a_dim)

        self.encoder_t = ResNet18(N, channels, self.dim_touch)  # ConvNet(N, channels, self.dim_touch)

        self.encoder_h_pos = nn.Sequential(nn.Linear(self.dim_touch, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, a_dim))
        self.encoder_h_rot = nn.Sequential(nn.Linear(self.dim_touch, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 1))

        eigen0 = torch.tensor(eigen_vals)

        if learn_var == 1:
            self.eigen = nn.Parameter(eigen0, requires_grad=True)  # 0.05
        else:
            self.eigen = nn.Parameter(eigen0, requires_grad=False)  # 0.05

    def forward(self, o):

        z = torch.sigmoid(self.encoder_z(o).view(-1, self.a_dim))
        touch = F.normalize(self.encoder_t(o), dim=-1).view(-1, self.dim_touch)

        mu_h = self.encoder_h_pos(touch).view(-1, self.n_objs, self.a_dim)
        h_theta = self.encoder_h_rot(touch).view(-1, self.n_objs, 1)

        if not self.rotations:
            h_theta *= 0
            var_h = torch.unsqueeze(torch.unsqueeze(torch.diag(self.eigen), 0), 1).repeat(mu_h.shape[0], 1, 1, 1)
        else:
            rot11 = torch.cos(h_theta)
            rot12 = -torch.sin(h_theta)
            rot21 = torch.sin(h_theta)
            rot22 = torch.cos(h_theta)
            rot = torch.cat([torch.cat([rot11, rot12], -1), torch.cat([rot21, rot22], -1)], 1)
            var_h = torch.unsqueeze(torch.diag_embed(self.eigen ** 2 + 1e-5), 0)
            var_h = torch.unsqueeze(rot.transpose(-2, -1) @ var_h @ rot, 1)

        if self.rl:
            return torch.cat([z, mu_h[:, 0]], 1)

        return z, mu_h, var_h, h_theta, touch


class Policy(nn.Module):

    def __init__(self, Representation, detach, z_dim, a_dim, a_max):
        super().__init__()

        self.a_dim = a_dim
        self.a_max = a_max
        self.detach = detach
        hidden_fc = 64

        self.body = Representation

        self.actor = nn.Sequential(nn.Linear(z_dim, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, a_dim * 2))
        self.value = nn.Sequential(nn.Linear(z_dim, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, 1))

    def forward(self, x):

        h = self.body(x) if self.body is not None else x
        h = h.detach() if self.detach else h

        policy = self.actor(h)
        v = self.value(h)
        mu = torch.tanh(policy[:, 0:self.a_dim]) * self.a_max
        sigma = torch.sigmoid(policy[:, self.a_dim:2 * self.a_dim]) * 1 + 0.0001
        v = v[:, -1]

        return mu, sigma, v


class ActorCritic(nn.Module):

    def __init__(self, Representation, detach, z_dim, a_dim, a_max):
        super(ActorCritic, self).__init__()

        self.network = Policy(Representation, detach, z_dim, a_dim, a_max)

    def forward(self, st):

        mu, sigma, v = self.network(st)

        return mu, sigma, v

    def get_action(self, st, test=False):

        mu, sigma, v = self.forward(st)

        if test:
            return mu, None, None

        m = MultivariateNormal(mu, torch.diag_embed(sigma))
        a = m.sample()
        logprob = m.log_prob(a)
        H = m.entropy()

        return a, logprob, sigma.detach().cpu().numpy()

    def evaluate(self, st, at):

        mu, sigma, v = self.forward(st)

        m = MultivariateNormal(mu, torch.diag_embed(sigma))
        logprob = m.log_prob(at)
        H = m.entropy()

        return logprob, v, H

import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

from networks import phi, VAE
from transporter import FeatureEncoder, PoseRegressor, RefineNet, Transporter
from utils import otsu as otsu_algorithm


class Haptic_Repr:

    def __init__(self, dataset, N, channels, segment, a_dim, frq, learn_var, rotations, eigen_vals, device):
        super(Haptic_Repr, self).__init__()

        self.dataset = dataset

        self.device = device
        self.N = N
        self.segment = segment
        self.learn_var = learn_var

        self.frq = frq
        self.rotations = rotations
        self.tau_gb = 0.01
        self.tau_lc = 0.1
        self.weight_contr = 1
        self.use_otsu = 1

        self.net = phi(N, channels, a_dim, 1, learn_var, rotations, eigen_vals, device).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


    def fit_Rep(self, batch, epoch_idx, epoch, train=True):

        ot, ot_2d, ot1, at, pos, next_pos = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        ot = ot.to(self.device)
        ot_2d = ot_2d.to(self.device)
        ot1 = ot1.to(self.device)
        at = at.to(self.device)
        pos = pos.to(self.device)
        next_pos = next_pos.to(self.device)

        zt, mu_ht, var_ht, theta_ht, touch_t = self.net(ot)
        zt1, mu_ht1, var_ht1, theta_ht1, touch_t1 = self.net(ot1)

        gauss = MultivariateNormal(mu_ht, var_ht)
        gauss1 = MultivariateNormal(mu_ht1, var_ht1)

        ''' EQUIVARIANCE '''
        loss_Eqv = torch.mean(torch.sum(((zt1 - zt) - at) ** 2, -1))

        ''' INJECTIVITY '''

        rnd_idx = np.arange(mu_ht.shape[0])
        np.random.shuffle(rnd_idx)

        dist_touch = torch.sum((torch.unsqueeze(touch_t[rnd_idx], 1) - touch_t1.detach()) ** 2, -1) / self.tau_gb

        pos_d = torch.sum((touch_t - touch_t1.detach()) ** 2, -1) / self.tau_gb
        neg_elem_d = dist_touch
        neg_d = torch.logsumexp(-neg_elem_d, 0)
        loss_contrastive = torch.mean(pos_d + neg_d)

        ''' POSITIVE HAPTIC '''
        n_points = 100
        z_segment_set = torch.rand(zt.shape[0], 1, n_points).to(self.device) * torch.unsqueeze((zt1 - zt), -1) + torch.unsqueeze(zt, -1)
        pos_nll = torch.cat([-gauss.log_prob(z_segment_set[:, :, j].view(mu_ht.shape).detach()) for j in range(n_points)], -1)
        pos_haptic = torch.min(pos_nll, -1)[0]

        ''' NEGATIVE HAPTIC '''
        neg_haptic = torch.mean(0.5 * (kl_divergence(gauss, gauss1) + kl_divergence(gauss1, gauss)), dim=1)

        err_inv = torch.sum((touch_t - touch_t1) ** 2, -1).detach()
        neg_idx = torch.argsort(err_inv).detach()

        if self.use_otsu == 1:
            neg_max_idx = int(otsu_algorithm(np.sort(err_inv.detach().cpu().numpy())))
        else:
            neg_max_idx = int(pos_haptic.shape[0] * (1 - self.frq))

        loss_haptic_pos = pos_haptic[neg_idx[neg_max_idx:]]
        loss_haptic_neg = neg_haptic[neg_idx[:neg_max_idx]]

        loss_haptic = 1.*torch.mean(loss_haptic_pos) + 1.*torch.mean(loss_haptic_neg)

        ''' LOSS OPT '''
        loss = loss_Eqv + torch.mean(loss_contrastive) * self.weight_contr + loss_haptic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ''' LOGS '''
        vec_pos = pos[:, 0] - pos[:, 1]
        vec_z = zt - mu_ht[:, 0]
        avg_rel_err = torch.mean(torch.sum((vec_pos - vec_z) ** 2, -1)).detach().cpu().item()

        metrics_logs = {'avg_isometry_err': avg_rel_err}

        return metrics_logs

    def get_rep(self, ot, ot1):

        zt, mu_ht, logvar_ht, theta_ht, touch_t = self.net(ot)

        return zt.detach(), mu_ht.detach(), logvar_ht.detach(), theta_ht.detach()

    def save_model(self, path_dir, exp_name):
        fname_psi = path_dir+exp_name+".mdl"
        torch.save(self.net.state_dict(), fname_psi)

    def load_model(self, path_dir, exp_name):
        fname_psi = path_dir+exp_name+".mdl"
        state_dict_psi = torch.load(fname_psi, map_location=self.device)
        state_dict_psi['eigen'] = self.net.eigen
        self.net.load_state_dict(state_dict_psi, strict=False)
        self.net.eval()


class Transporter_Repr:

    def __init__(self, dataset, k, channels, device):
        super(Transporter_Repr, self).__init__()

        self.dataset = dataset
        self.learn_var = 0

        self.device = device

        feature_encoder = FeatureEncoder(channels)
        pose_regressor = PoseRegressor(channels, k)
        refine_net = RefineNet(channels)

        self.net = Transporter(feature_encoder, pose_regressor, refine_net).to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def fit_Rep(self, batch, i=None, e=None, train=True):
        ot, ot_2d, ot1, at, pos, next_pos = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        st = ot.to(self.device)
        st1 = ot1.to(self.device)
        at = at.to(self.device)
        pos = pos.to(self.device)
        next_pos = next_pos.to(self.device)

        st1_hat, zt = self.net(st, st1)

        loss = self.net.get_loss(st1, st1_hat)

        avg_loss = loss.detach().cpu().item()
        vec_pos = pos[:, 0] - pos[:, 1]
        vec_z = zt[:, 0] - zt[:, 1]
        avg_rel_err = torch.mean(torch.sum((vec_pos - vec_z) ** 2, -1)).detach().cpu().item()

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics_logs = {'avg_isometry_err': avg_rel_err}

        return metrics_logs

    def get_rep(self, ot, ot1):

        ot1_hat, zt = self.net(ot, ot1)

        return zt.detach(), None, None, None

    def save_model(self, path_dir, exp_name):
        fname_psi = path_dir + exp_name + ".mdl"
        torch.save(self.net.state_dict(), fname_psi)

    def load_model(self, path_dir, exp_name):
        fname_psi = path_dir + exp_name + ".mdl"
        state_dict_psi = torch.load(fname_psi, map_location=self.device)
        self.net.load_state_dict(state_dict_psi)
        self.net.eval()


class VAE_Repr:

    def __init__(self, N, z_dim, h_features, channels, vae_equi, device):
        super(VAE_Repr, self).__init__()

        self.device = device
        self.vae_equi = vae_equi

        self.net = VAE(N, channels, z_dim, h_features).to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def fit_Rep(self, batch, train=True):
        ot, ot_2d, ot1, at, pos, next_pos = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        st = ot.to(self.device)
        st1 = ot1.to(self.device)
        at = at.to(self.device)
        pos = pos.to(self.device)
        next_pos = next_pos.to(self.device)

        z1, mu1, logvar1, s_hat1 = self.net(st)
        z2, mu2, logvar2, s_hat2 = self.net(st1)

        loss_mse1, loss_kl1 = self.net.get_vae_loss(st, s_hat1, mu1, logvar1)
        loss_mse2, loss_kl2 = self.net.get_vae_loss(st1, s_hat2, mu2, logvar2)
        loss_eq = torch.mean(torch.sum((z2 - z1 - at) ** 2, -1))

        loss = loss_mse1 + loss_kl1 + loss_mse2 + loss_kl2 + self.vae_equi*loss_eq

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = loss.detach().cpu().item()
        avg_mse_loss = (loss_mse1 + loss_mse2).detach().cpu().item()
        avg_kl_loss = (loss_kl1 + loss_kl2).detach().cpu().item()
        avg_eq_loss = loss_eq.detach().cpu().item()
        vec_pos = pos[:, 0] - pos[:, 1]
        vec_z = z1 - mu1
        avg_rel_err = torch.mean(torch.sum((vec_pos - vec_z) ** 2, -1)).detach().cpu().item()

        metrics_logs = {'avg_isometry_err': avg_rel_err}

        return metrics_logs

    def get_rep(self, ot, ot1):
        z1, mu1, logvar1, s_hat1 = self.net(ot)
        return z1.detach(), mu1.detach(), logvar1.detach()

    def save_model(self, path_dir, exp_name):
        fname_psi = path_dir + exp_name + ".mdl"
        torch.save(self.net.state_dict(), fname_psi)

    def load_model(self, path_dir, exp_name):
        fname_psi = path_dir + exp_name + ".mdl"
        state_dict_psi = torch.load(fname_psi, map_location=self.device)
        self.net.load_state_dict(state_dict_psi)
        self.net.eval()




































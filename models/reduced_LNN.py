import torch
import torch.nn as nn

from models.geom_LNN import LNN
from models.c_autoencoder import ConstrainedAE


class RO_LNN(nn.Module):
    """
    Implementation of a reduced-order LNN:
    for joint training of structure-preserving state-space embedding & latent LNN.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.n_dof_lat = kwargs.get('layer_dims_AE', [1, 1])[-1]
        self.n_dof_full = kwargs.get('layer_dims_AE', [1, 1])[0]
        self.autoencoder = ConstrainedAE(**kwargs)
        self.lagrangianNN = LNN(self.n_dof_lat, **kwargs)
        self.lagrangianNN.latent_flag = True
        self.device = kwargs.get('device', 'cpu')

    def forward(self, x, u_in):

        k_steps = u_in.shape[1]

        # reduce gt states
        q, dq = torch.split(x, int(self.n_dof_full), dim=-1)
        q_check, dq_check = self.autoencoder.encoder_vel(q.view(-1, self.n_dof_full), dx=dq.view(-1, self.n_dof_full))
        q_check, dq_check = q_check.view(-1, k_steps, self.n_dof_lat), dq_check.view(-1, k_steps, self.n_dof_lat)
        x_check = torch.cat((q_check, dq_check), dim=-1)

        # LNN predictions from the initial conditions
        self.lagrangianNN.ae = self.autoencoder
        x_check_pred = self.lagrangianNN(x_check, u_in)

        # decode the LNN predictions
        q_check_pred, dq_check_pred = torch.split(x_check_pred, int(self.n_dof_lat), dim=-1)
        q_pred_hat, dq_pred_hat = self.autoencoder.decoder_vel(q_check_pred.view(-1, self.n_dof_lat), dz=dq_check_pred.view(-1, self.n_dof_lat))
        q_pred_hat, dq_pred_hat = q_pred_hat.view(-1, k_steps, self.n_dof_full), dq_pred_hat.view(-1, k_steps, self.n_dof_full)
        x_pred_hat = torch.cat((q_pred_hat, dq_pred_hat), dim=-1)

        # AE reconstruction only
        q_hat, dq_hat = self.autoencoder.decoder_vel(q_check.view(-1, self.n_dof_lat), dz=dq_check.view(-1, self.n_dof_lat))
        q_hat = q_hat.view(-1, k_steps, self.n_dof_full)
        dq_hat = dq_hat.view(-1, k_steps, self.n_dof_full)
        x_hat = torch.cat((q_hat, dq_hat), dim=-1)


        # returns reconstructed LNN pred, AE_encoded, latent LNN prediction, AE_only reconstruction
        return x_pred_hat, x_check, x_check_pred, x_hat

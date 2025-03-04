import numpy as np
import torch
import torch.nn as nn

import geoopt.tensor as gt
from utils.biorthogonal_manifold import BiOrthogonal


class ProjectionLayerPair(nn.Module):
    """
    Produce a layer-pair: encoder layer from N to r, with r<N, and left inverse of corresponding decoder layer.
    """
    def __init__(self, input_size, output_size, alpha=np.pi / 8, device=None):
        super().__init__()

        self._dim_N = input_size
        self._dim_r = output_size
        self._alpha_act = alpha
        if self._alpha_act > np.pi / 4 or self._alpha_act < 0:
            self._alpha_act = np.pi / 8

        self._eye_N = torch.eye(self._dim_N, dtype=torch.float32).view(self._dim_N, self._dim_N).to(device)
        self._eye_r = torch.eye(self._dim_r, dtype=torch.float32).view(self._dim_r, self._dim_r).to(device)
        self._zeros_r = torch.zeros(self._dim_r, dtype=torch.float32).to(device)

        self._a = (1 / np.sin(self._alpha_act)) ** 2 - (1 / np.cos(self._alpha_act)) ** 2
        self._b = (1 / np.sin(self._alpha_act)) ** 2 + (1 / np.cos(self._alpha_act)) ** 2
        self._t1 = self._b / self._a
        self._t2 = np.sqrt(2) / (self._a * np.sin(self._alpha_act))
        self._t3 = 1 / self._a
        self._t4 = 2 / (np.sin(self._alpha_act) * np.cos(self._alpha_act))
        self._t5 = np.sqrt(2) / np.cos(self._alpha_act)
        self._t6 = 2 * self._a

        self.biorthogonal = BiOrthogonal(self._dim_N, self._dim_r, dev=device)
        self.weight = gt.ManifoldParameter(self.biorthogonal.random(self._dim_N, self._dim_r, device=device), manifold=self.biorthogonal,requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(self._dim_N).to(device))

    def update_weight_params(self):
        self.DEC, self.X = self.biorthogonal._split_representation(self.weight)
        self.ENC = self.X.transpose(-1,-2)

    def enc_activation(self, x):
        return self._t1 * x + self._t2 - self._t3 * torch.sqrt((self._t4 * x + self._t5) ** 2 + self._t6)

    def dec_activation(self, x):
        return self._t1 * x - self._t2 + self._t3 * torch.sqrt((self._t4 * x - self._t5) ** 2 + self._t6)

    def d_enc_activation(self, x):
        return self._t1 - self._t3 * self._t4 * (self._t4 * x + self._t5) / torch.sqrt(
            (self._t4 * x + self._t5) ** 2 + self._t6)

    def d_dec_activation(self, x):
        return self._t1 + self._t3 * self._t4 * (self._t4 * x - self._t5) / torch.sqrt(
            (self._t4 * x - self._t5) ** 2 + self._t6)

    def fw_encoder_layer(self, x):
        a = self.ENC @ (x - self.bias).unsqueeze(-1)
        z = self.enc_activation(a.squeeze(-1))
        return z

    def fw_decoder_layer(self, z):
        x = (self.DEC @ self.dec_activation(z).unsqueeze(-1)).squeeze(-1) + self.bias
        return x

    def fw_encoder_layer_vel(self, x, x_prime):
        a = self.ENC @ (x - self.bias).unsqueeze(-1)
        z = self.enc_activation(a.squeeze(-1))
        jac = self.d_enc_activation(a) * self.ENC @ x_prime
        return z, jac

    def fw_decoder_layer_vel(self, z, x_prime):
        x = (self.DEC @ self.dec_activation(z).unsqueeze(-1)).squeeze(-1) + self.bias
        jac = self.DEC @ (self.d_dec_activation(z.unsqueeze(-1)) * x_prime)
        return x, jac

class ConstrainedAE(nn.Module):
    """
    Autoencoder network with pairwise biorthogonal layer pairs, encoder being a left inverse to the decoder.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")

        self.layer_dims = kwargs.get("layer_dims_AE", [1, 1])
        self.b0 = kwargs.get("b_init_AE", 0)
        self.alpha = kwargs.get("alpha_act_AE", np.pi / 10)
        self.x_eq = kwargs.get("equilibrium_point", torch.zeros(self.layer_dims[0], dtype=torch.float32).to(self.device))

        self.layers = nn.ModuleList()
        for i in range(1, len(self.layer_dims)):
            self.layers.append(
                ProjectionLayerPair(self.layer_dims[i - 1], self.layer_dims[i], self.alpha, self.device))
            torch.nn.init.constant_(self.layers[-1].bias, self.b0)
        self.update_parameters()

    def encoder_pos(self, x):
        z = self.layers[0].fw_encoder_layer(x)
        for i in range(1, len(self.layers)):
            z = self.layers[i].fw_encoder_layer(z)
        return z
    def decoder_pos(self, z):
        x_hat = self.layers[-1].fw_decoder_layer(z)
        for i in range(len(self.layers) - 2, -1, -1):
            x_hat = self.layers[i].fw_decoder_layer(x_hat)
        return x_hat

    def encoder_vel(self, x, dx=None, return_jac=False):
        z, jac_enc = self.layers[0].fw_encoder_layer_vel(x, self.layers[0]._eye_N)
        for i in range(1, len(self.layers)):
            z, jac_enc = self.layers[i].fw_encoder_layer_vel(z, jac_enc)
        if not return_jac:
            dz = (jac_enc @ dx.unsqueeze(-1)).squeeze(-1)
            return z, dz
        else:
            return jac_enc

    def decoder_vel(self, z, dz=None, return_jac=False):
        x_hat, jac_dec = self.layers[-1].fw_decoder_layer_vel(z, self.layers[-1]._eye_r)
        for i in range(len(self.layers) - 2, -1, -1):
            x_hat, jac_dec = self.layers[i].fw_decoder_layer_vel(x_hat, jac_dec)
        if not return_jac:
            dx_hat = (jac_dec @ dz.unsqueeze(-1)).squeeze(-1)
            return x_hat, dx_hat
        else:
            return jac_dec

    def forward_vel(self, x, dx):
        z, dz = self.encoder_vel(x, dx=dx)
        x_hat, dx_hat = self.decoder_vel(z, dz=dz)
        return x_hat, dx_hat

    def reduce_act(self, z, tau):
        jac_dec = self.decoder_vel(z, return_jac=True)
        tau_red = (jac_dec.transpose(-1, -2) @ tau.unsqueeze(-1)).squeeze(-1)
        return tau_red


    def update_parameters(self):
        for layer in self.layers:
            layer.update_weight_params()
        return None




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import geoopt as go
import geoopt.tensor as gt
from torchdiffeq import odeint

from utils.spd_manifold import SPD as SPDManifold
import utils.autodiff as numdiff


class SPDExpLayer(nn.Module):
    """
    converts an m=n(n+1)/2 dimensional vector input to symmetric matrix, applies exponential map to obtain SPD matrix.
    """
    def __init__(self, n_dof, parametrized_expmap=False, device=None):
        super().__init__()
        self.device = device
        self.n_dof = n_dof
        self.m = int((self.n_dof ** 2 + self.n_dof) / 2)

        # init geom stuff
        self.man = SPDManifold()
        self.zero_mat = torch.zeros(self.n_dof, self.n_dof).to(device)

        self.tril_indices = torch.tril_indices(self.n_dof, self.n_dof, offset=-1, device=device)
        self.diag_indices = torch.arange(self.n_dof, device=device)

        self.parametrized_expmap = parametrized_expmap
        if self.parametrized_expmap:
            self.exp_point = gt.ManifoldParameter(self.man.origin(self.n_dof, self.n_dof, device=device), manifold=self.man, requires_grad=True)
        else:
            self.exp_point = torch.eye(self.n_dof).to(device)
            self.der_zeros = torch.zeros(self.n_dof, self.n_dof, int((self.n_dof ** 2 + self.n_dof) / 2), dtype=torch.float32, device=self.device)

            # derivative of rearranging
            self.dUdx = self.der_zeros
            self.dUdx[self.diag_indices, self.diag_indices, self.diag_indices] = 1
            self.dUdx[self.tril_indices[1], self.tril_indices[0], torch.arange(self.m - self.n_dof) + self.n_dof] = 1
            self.dUdx[self.tril_indices[0], self.tril_indices[1], torch.arange(self.m - self.n_dof) + self.n_dof] = 1
            self.dUdx = self.dUdx.reshape(self.n_dof**2, self.m).unsqueeze(0)


    def forward(self, x, dx):

        U = self.vec2symmat(x)

        if self.parametrized_expmap:
            out = self.man.expmap(self.exp_point, U)
            md = numdiff.matrix_vec_diff(out, x)
            dout = md @ dx.unsqueeze(1)
        else:
            out = torch.matrix_exp(U)
            e, v = torch.linalg.eigh(U, 'L')

            P = self.batch_kron(v, v)
            R = self.comp_R(e)
            D = torch.diag_embed(R.transpose(-1, -2).reshape(R.shape[0], -1))
            der = P @ D @ P.transpose(-1, -2)
            der = der @ self.dUdx
            der = der.view(-1, self.n_dof, self.n_dof, self.m).transpose(1, 2)
            dout = der @ dx.unsqueeze(1)

        return out, dout
    def batch_kron(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        b = A.shape[0]
        kron_prod = A.view(b, self.n_dof, 1, self.n_dof, 1) * B.view(b, 1, self.n_dof, 1, self.n_dof)

        return kron_prod.view(b, self.n_dof**2, self.n_dof**2)
    def comp_R(self, b):

        B_diff = b.unsqueeze(2) - b.unsqueeze(1)
        exp_b = torch.exp(b)
        exp_diff = exp_b.unsqueeze(2) - exp_b.unsqueeze(1)
        non_zero_mask = B_diff != 0
        R = torch.zeros_like(B_diff)
        R[non_zero_mask] = exp_diff[non_zero_mask] / B_diff[non_zero_mask]
        R[:, self.diag_indices, self.diag_indices] = exp_b

        return R

    def vec2symmat(self, vec):

        bs = vec.shape[0]
        if len(vec.shape) == 2:
            symmat = self.zero_mat.repeat(bs, 1, 1)
        elif len(vec.shape) == 3:
            symmat = self.zero_mat.view(1, self.n_dof, self.n_dof, 1).repeat(bs, 1, 1, vec.shape[2])
        symmat[:, self.diag_indices, self.diag_indices] = vec[:, :self.n_dof]
        symmat[:, self.tril_indices[1], self.tril_indices[0]] = vec[:, self.n_dof:]
        symmat[:, self.tril_indices[0], self.tril_indices[1]] = vec[:, self.n_dof:]

        return symmat

class EuclideanLayer(nn.Module):
    """
    Standard Euclidean fully connected layer
    """

    def __init__(self, n_in, n_width, activation="ReLu", device="cpu"):
        super().__init__()

        self.man = go.manifolds.Euclidean(ndim=0)
        self.weight = gt.ManifoldParameter(torch.Tensor(n_width, n_in).to(device), manifold=self.man, requires_grad=True)
        self.bias = gt.ManifoldParameter(torch.Tensor(n_width).to(device), manifold=self.man, requires_grad=True)
        self.activation = activation

        if self.activation == "SoftPlus":
            self.softplus_beta = int(1.0)
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.dg = self.d_Softplus
        elif self.activation == "Linear":
            self.g = nn.Identity()
            self.dg = self.d_Linear

    def forward(self, q, dq):
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        dout = (self.dg(a).unsqueeze(-1) * self.weight) @ dq
        return out, dout

    def forward_nodiff(self, q):
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        return out

    def d_Linear(self, x):
        return torch.full_like(x, 1)

    def d_Softplus(self, x):
        clamped_x = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self.softplus_beta * clamped_x)
        out = exp_x / (exp_x + 1.0)
        return out


class LNN(nn.Module):
    """
    Geometric LNN, with kinetic energy network on SPD-tangent space.
    """
    def __init__(self, n_dof, **kwargs):
        super().__init__()
        # DoF of system
        self.n_dof = n_dof
        self.latent_flag = False
        self.detach = kwargs.get('detach', True)
        self.n_jac_entries = kwargs.get('n_jac_entries', 0)

        # euclidean hidden layers: kinetic & potential energy NN dimensions
        self.n_width_T_eucl = kwargs.get('n_width_T_eucl', 64)
        self.n_depth_T_eucl = kwargs.get("n_depth_T_eucl", 2)
        self.n_width_V = kwargs.get('n_width_V_eucl', 64)
        self.n_depth_V = kwargs.get("n_depth_V_eucl", 2)

        self.activation = kwargs.get("activation", "SoftPlus")
        self.gain_hidden = kwargs.get("gain_hidden", np.sqrt(2.))

        # if true, exp wrt a weight, if false it's wrt Identity
        self.do_exp_wrt_p = kwargs.get('exp_wrt_p', False)

        # for ode int
        self.ode_delta_t = kwargs.get('ode_timestep', -1.)
        self.ode_solver = kwargs.get('ode_solver', 'euler')
        self.tau_in_vec = None
        self.time_vec = None

        # for computations
        self.device = kwargs.get("device", "cpu")
        self.eye_mat = torch.eye(self.n_dof).reshape(1, self.n_dof, self.n_dof).to(self.device)
        self.zero_mat = torch.zeros(self.n_dof, self.n_dof).reshape(1, self.n_dof, self.n_dof).to(self.device)

        # SPD manifold dimension
        self.m = int((self.n_dof ** 2 + self.n_dof) / 2)

        # for storage
        self.out_E_kin = torch.zeros(0, )
        self.out_E_pot = torch.zeros(0, )
        self.out_M = torch.zeros(0, self.n_dof, self.n_dof)
        self.out_c = torch.zeros(0, self.n_dof)
        self.out_g = torch.zeros(0, self.n_dof)

        self.ae = None

        def init_layer(layer, is_out=False):
            if is_out:
                torch.nn.init.xavier_normal_(layer.weight, 1)
            else:
                if self.gain_hidden <= 0.0:
                    if layer.activation == "SoftPlus":
                        gain = np.sqrt(2. / (1 + np.pi ** 2 / 6))
                else:
                    gain = self.gain_hidden

                torch.nn.init.xavier_normal_(layer.weight, gain)
            torch.nn.init.constant_(layer.bias, 0)

        # construct kinetic energy network layers
        self.layers_M = nn.ModuleList()

        # 1st part: normal euclidean network of desired dimensions
        self.layers_M.append(EuclideanLayer(self.n_dof, self.n_width_T_eucl, activation=self.activation, device=self.device))
        init_layer(self.layers_M[-1])
        for _ in range(1, self.n_depth_T_eucl):
            self.layers_M.append(
                EuclideanLayer(self.n_width_T_eucl, self.n_width_T_eucl, activation=self.activation, device=self.device))
            init_layer(self.layers_M[-1])
        self.layers_M.append(EuclideanLayer(self.n_width_T_eucl, self.m, activation="Linear", device=self.device))
        init_layer(self.layers_M[-1], is_out=True)

        # expmap-layer
        self.layers_M.append(SPDExpLayer(self.n_dof, parametrized_expmap=self.do_exp_wrt_p, device=self.device))

        # potential energy network
        self.layers_V = nn.ModuleList()
        self.layers_V.append(EuclideanLayer(self.n_dof, self.n_width_V, activation=self.activation, device=self.device))
        init_layer(self.layers_V[-1])
        for _ in range(1, self.n_depth_V):
            self.layers_V.append(
                EuclideanLayer(self.n_width_V, self.n_width_V, activation=self.activation,
                               device=self.device))
            init_layer(self.layers_V[-1])
        self.layers_V.append(EuclideanLayer(self.n_width_V, 1, activation="Linear", device=self.device))
        init_layer(self.layers_V[-1], is_out=True)


    def dynamic_model(self, q, dq):

        y, dy_dq = self.layers_M[0](q, self.eye_mat)
        # tangent space network: n layers + 1 output layer
        for i in range(1, self.n_depth_T_eucl+1):
            y, dy_dq = self.layers_M[i](y, dy_dq)
        M, dMdq = self.layers_M[-1](y, dy_dq)

        # coriolis
        C = 0.5 * torch.einsum('bijk,bk->bij', (dMdq + dMdq.transpose(2, 3) - dMdq.transpose(1, 3)), dq)
        c = (C @ dq.unsqueeze(-1)).squeeze(-1)

        # kinetic energy
        T = 1. / 2. * (dq.view(-1, 1, self.n_dof) @ M @ dq.view(-1, self.n_dof, 1)).view(-1)

        # potential: energy & forces
        V, der_V = self.layers_V[0](q, self.eye_mat)
        for i in range(1, len(self.layers_V)):
            V, der_V = self.layers_V[i](V, der_V)
        V = V.squeeze(-1)
        g = der_V.squeeze(-2)

        return M, c, g, T, V

    def fw_prediction(self, q, dq, tau, save_outputs=True):

        M, c, g, T, V = self.dynamic_model(q, dq)

        M_inv = torch.inverse(M)

        ddq_pred = torch.matmul(M_inv, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)

        if save_outputs:
            if self.out_M.shape[0] == 0:
                self.out_M = M.unsqueeze(1)
                self.out_c = c.unsqueeze(1)
                self.out_g = g.unsqueeze(1)
                self.out_E_kin = T.unsqueeze(1)
                self.out_E_pot = V.unsqueeze(1)
            else:
                self.out_M = torch.cat((self.out_M, M.unsqueeze(1)), dim=1)
                self.out_c = torch.cat((self.out_c, c.unsqueeze(1)), dim=1)
                self.out_g = torch.cat((self.out_g, g.unsqueeze(1)), dim=1)
                self.out_E_kin = torch.cat((self.out_E_kin, T.unsqueeze(1)), dim=1)
                self.out_E_pot = torch.cat((self.out_E_pot, V.unsqueeze(1)), dim=1)

        return ddq_pred

    def forward(self, x, u, do_ode=True):

        self.reset_outs_and_integrators()

        if do_ode:
            self.tau_in_vec = u
            self.last_saved_t = -1.
            out = odeint(self.ode_fun, x[:, 0], self.ode_time_vec, method=self.ode_solver)
            x_hat_at_ode = torch.transpose(out, 0, 1)
            x_hat = self.get_x_hat_at_t(x_hat_at_ode)
            return x_hat
        else:
            q, dq = torch.split(x, split_size_or_sections=int(self.n_dof), dim=-1)
            ddq = self.fw_prediction(q, dq, u, save_outputs=True)
            dx = torch.cat((dq, ddq), dim=-1)
            return dx


    def ode_fun(self, t, x):

        if torch.any(torch.abs(self.time_vec - t) < 1e-6) and t!=self.last_saved_t:
            save_out = True
            self.last_saved_t = t
        else:
            save_out = False

        q, dq = torch.split(x, split_size_or_sections=int(self.n_dof), dim=-1)
        ddq = self.fw_prediction(q, dq, tau=self.input_interpolation(t, q), save_outputs=save_out)
        dx = torch.cat((dq, ddq), dim=-1)

        return dx

    def set_solver_times(self, t):

        t = t.to(self.device)
        self.time_vec = (t - t[:, 0].unsqueeze(-1))[0].squeeze().to(self.device)  # (k+1, 1)
        self.delta_time_vec = round(float(torch.mean(self.time_vec[1:] - self.time_vec[:-1])), 6)

        time_horizon = self.time_vec[-1].item()
        if self.ode_delta_t <= 0:
            self.ode_delta_t = round(float(time_horizon / (t.shape[1] - 1)), 6)
        addition = max(int(round(self.delta_time_vec/self.ode_delta_t)), 1)

        self.ode_time_vec = torch.linspace(0, (time_horizon + addition*self.ode_delta_t), round((time_horizon + self.ode_delta_t) / self.ode_delta_t) + addition).to(self.device)  # size (k)

        return None

    def input_interpolation(self, t_interp, q):
        # given an input u, it interpolates & also converts to (reduced) tau_ctrl
        ind_interp = torch.argmin(torch.abs(self.time_vec - t_interp)).item()
        tau_interp = self.tau_in_vec[:, ind_interp, :]
        if self.latent_flag:
            tau_ctrl = self.ae.reduce_act(q, tau_interp)
            return tau_ctrl
        else:
            return tau_interp

    def get_x_hat_at_t(self, x_hat_at_ode):
        diff = torch.abs(self.ode_time_vec[:, None] - self.time_vec[None, :])
        mask = torch.any(diff <= 1e-6, dim=1)
        x_hat = x_hat_at_ode[:, mask, :]
        return x_hat

    def reset_outs_and_integrators(self):
        self.tau_in_vec = None
        self.out_E_kin = torch.zeros(0, )
        self.out_E_pot = torch.zeros(0, )
        self.out_M = torch.zeros(0, self.n_dof, self.n_dof)
        self.out_c = torch.zeros(0, self.n_dof)
        self.out_g = torch.zeros(0, self.n_dof)

        return None
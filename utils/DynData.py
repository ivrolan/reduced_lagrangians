import pickle
import torch
from torch.utils.data import Dataset

class DynDataset(Dataset):

    def __init__(self, n_dof, file_path, n_rec=[], phase='train', sampling_mode='random', n_samples=300, n_delta_t=1, device=None):
        self.device = device

        self.n_dof = n_dof
        self.phase = phase
        self.sampling_mode = sampling_mode
        self.n_delta_t = n_delta_t
        self.n_samples = n_samples

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # in main code, n_rec is either [], [no_recs], or [from, to]
        if len(n_rec) == 0:
            n_rec = [0, data['t'].shape[0]-1]
            self.no_recs = data['t'].shape[0]
        elif len(n_rec) == 1:
            self.no_recs = n_rec
            idx_rec = torch.randint(0, 10, ()).item()
            n_rec = [idx_rec, idx_rec+self.no_recs]
        else:
            self.no_recs = n_rec[1] - n_rec[0] + 1
        self.rec_start = n_rec[0]
        self.rec_end = n_rec[1]

        self.t = torch.tensor(data['t'][n_rec[0]:n_rec[1] + 1]).float().to(self.device).view(self.no_recs, -1)
        self.q = torch.tensor(data['q'][n_rec[0]:n_rec[1] + 1], requires_grad=False).float().to(self.device).view(self.no_recs, -1, n_dof)
        self.dq = torch.tensor(data['dq'][n_rec[0]:n_rec[1] + 1]).float().to(self.device).view(self.no_recs, -1, n_dof)
        self.ddq = torch.tensor(data['ddq'][n_rec[0]:n_rec[1] + 1]).float().to(self.device).view(self.no_recs, -1, n_dof)
        self.tau = torch.tensor(data['tau'][n_rec[0]:n_rec[1] + 1]).float().to(self.device).view(self.no_recs, -1, n_dof)

        self.e_kin = torch.tensor(data['e_kin'][n_rec[0]:n_rec[1] + 1]).float().to(self.device).view(self.no_recs, -1)
        self.e_pot = torch.tensor(data['e_pot'][n_rec[0]:n_rec[1] + 1]).float().to(self.device).view(self.no_recs, -1)

        # adjust for sample rate
        t_len = self.t.shape[-1]
        self.t = self.t[:, 0:t_len:self.n_delta_t].contiguous()
        self.q = self.q[:, 0:t_len:self.n_delta_t].contiguous()
        self.dq = self.dq[:, 0:t_len:self.n_delta_t].contiguous()
        self.ddq = self.ddq[:, 0:t_len:self.n_delta_t].contiguous()
        self.tau = self.tau[:, 0:t_len:self.n_delta_t].contiguous()
        self.e_kin = self.e_kin[:, 0:t_len:self.n_delta_t].contiguous()
        self.e_pot = self.e_pot[:, 0:t_len:self.n_delta_t].contiguous()

        self.t_vec = self.t.view(-1)
        self.q_vec = self.q.view(-1, n_dof)
        self.dq_vec = self.dq.view(-1, n_dof)
        self.ddq_vec = self.ddq.view(-1, n_dof)
        self.tau_vec = self.tau.view(-1, n_dof)
        self.e_kin_vec = self.e_kin.view(-1)
        self.e_pot_vec = self.e_pot.view(-1)

        self.n_total_datapoints = self.t_vec.shape[0]

        self.sampling_idxs = torch.randperm(self.n_total_datapoints)[:int(min(self.n_samples, self.n_total_datapoints))]


    def __len__(self):
        if self.sampling_mode == 'uniform':
            len_data = round(self.n_total_datapoints)
        else:
            len_data = self.sampling_idxs.shape[0]
        return len_data

    def __getitem__(self, idx):

        if self.sampling_mode == 'uniform':
            s_idx = idx
        else:
            s_idx = self.sampling_idxs[idx]

        x = torch.cat((self.q_vec[s_idx], self.dq_vec[s_idx]), dim=-1)
        dx = torch.cat((self.dq_vec[s_idx], self.ddq_vec[s_idx]), dim=-1)
        tau = self.tau_vec[s_idx]
        t = self.t_vec[s_idx]
        e_kin = self.e_kin_vec[s_idx]
        e_pot = self.e_pot_vec[s_idx]

        return x, dx, tau, t, e_kin, e_pot


class DynDatasetODE(DynDataset):

    def __init__(self, n_dof, file_path, n_rec, h_step_horizon=1, phase='train', sampling_mode='random', n_samples=300, n_delta_t=1, device=None):
        super().__init__(n_dof, file_path, n_rec, phase, sampling_mode, n_samples, n_delta_t, device)
        if h_step_horizon == 'inf':
            self.h_steps = self.t.shape[1] - 1
        else:
            self.h_steps = int(h_step_horizon)  # number of prediction steps per sample

        self.t_list, self.q_list, self.dq_list, self.ddq_list, self.tau_list, self.u_list, self.e_kin_list, self.e_pot_list = [], [], [], [], [], [], [], []
        for i in range(0, self.no_recs):
            for j in range(0, self.t[i].shape[0] - self.h_steps):
                self.t_list.append(self.t[i, j:j + self.h_steps + 1])
                self.q_list.append(self.q[i, j:j + self.h_steps + 1])
                self.dq_list.append(self.dq[i, j:j + self.h_steps + 1])
                self.ddq_list.append(self.ddq[i, j:j + self.h_steps + 1])
                self.tau_list.append(self.tau[i, j:j + self.h_steps + 1])
                self.e_kin_list.append(self.e_kin[i, j:j + self.h_steps + 1])
                self.e_pot_list.append(self.e_pot[i, j:j + self.h_steps + 1])

        self.usable_datapoints = len(self.t_list)
        self.sampling_idxs = torch.randperm(self.usable_datapoints)[:int(min(self.n_samples, self.usable_datapoints))]

    def __len__(self):
        if self.sampling_mode == 'uniform':
            len_data = self.usable_datapoints
        else:
            len_data = self.sampling_idxs.shape[0]
        return len_data

    def __getitem__(self, idx):
        if self.sampling_mode == 'uniform':
            s_idx = idx
        else:
            s_idx = self.sampling_idxs[idx]

        x = torch.cat([self.q_list[s_idx], self.dq_list[s_idx]], dim=-1)
        dx = torch.cat([self.dq_list[s_idx], self.ddq_list[s_idx]], dim=-1)
        tau = self.tau_list[s_idx]
        t = self.t_list[s_idx]
        e_kin = self.e_kin_list[s_idx]
        e_pot = self.e_pot_list[s_idx]

        return x, dx, tau, t, e_kin, e_pot


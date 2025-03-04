import os, time

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('macosx')

import torch
from torch.utils.data import DataLoader

from geoopt.optim import RiemannianAdam

from models.geom_LNN import LNN
from utils.DynData import DynDataset, DynDatasetODE


def plot_pos_vel_pred(state_gt, state_pred, delta_t):
    n_dof = int(state_gt.shape[-1] / 2)
    n_trajs = state_gt.shape[0]
    n_len_traj = state_gt.shape[1]
    state_gt = state_gt.reshape(-1, 2*n_dof)
    state_pred = state_pred.reshape(-1, 2*n_dof)
    q_gt, dq_gt = np.split(state_gt, indices_or_sections=int(n_dof), axis=-1)
    q_pred, dq_pred = np.split(state_pred, indices_or_sections=int(n_dof), axis=-1)

    pos_mean = np.mean(np.linalg.norm(q_pred - q_gt, axis=1))
    pos_std = np.std(np.linalg.norm(q_pred - q_gt, axis=1))
    vel_mean = np.mean(np.linalg.norm(dq_pred - dq_gt, axis=1))
    vel_std = np.std(np.linalg.norm(dq_pred - dq_gt, axis=1))


    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f'position [rad] and velocity [rad/s]')
    for i in range(0, n_dof):
        ax_pos = fig.add_subplot(n_dof, 2, 2 * i + 1)
        ax_pos.plot(q_gt[:, i], color='k', label='ground truth')
        ax_pos.plot(q_pred[:, i], color='orange', label='LNN')
        ax_pos.set_ylabel(f'joint {i + 1}')

        ax_vel = fig.add_subplot(n_dof, 2, 2 * (i + 1))
        ax_vel.plot(dq_gt[:, i], color='k', label='ground truth')
        ax_vel.plot(dq_pred[:, i], color='orange', label='LNN')

        if i == 0:
            ax_pos.set_title(f'pos err mean:{pos_mean:.2e}, std:{pos_std:.2e}')
            ax_vel.set_title(f'vel err mean:{vel_mean:.2e}, std:{vel_std:.2e}')
            ax_vel.legend()

        ax_pos.set_xticks([])
        ax_vel.set_xticks([])
        tick_positions = np.array([0.0])
        tick_spacing = 4
        for n in range(1, n_trajs + 1):
            new_traj_pos = n * n_len_traj
            new_tick_pos = np.linspace(new_traj_pos - n_len_traj, new_traj_pos - 1, tick_spacing, dtype=int)
            tick_positions = np.append(tick_positions, new_tick_pos[1:])
            if n < n_trajs:
                ax_pos.axvline(x=new_traj_pos, color='lightgrey')
                ax_vel.axvline(x=new_traj_pos, color='lightgrey')
            if i == n_dof - 1:
                x_mid = 0.5 * (new_traj_pos + (n - 1) * n_len_traj)
                ax_pos.text(x_mid, ax_pos.get_ylim()[1] * 1.15, f'trajectory {n}', horizontalalignment='center',
                            color='black', fontsize=9)
                ax_vel.text(x_mid, ax_vel.get_ylim()[1] * 1.15, f'trajectory {n}', horizontalalignment='center',
                            color='black', fontsize=9)
        tick_labels = np.tile(np.linspace(0, delta_t * n_len_traj, tick_spacing)[1:], n_trajs)
        tick_labels = np.insert(tick_labels, 0, 0.0)
        if i == n_dof - 1:
            ax_pos.set_xticks(tick_positions)
            ax_pos.set_xticklabels([f"{x:.2f}" for x in tick_labels])
            ax_vel.set_xticks(tick_positions)
            ax_vel.set_xticklabels([f"{x:.2f}" for x in tick_labels])
            ax_pos.set_xlabel(f'time [s]')
            ax_vel.set_xlabel(f'time [s]')

    return None

def train_epoch(dataloader, model, optimizer, ode=True, phase='train'):
    epoch_loss = 0.0
    pred_err = torch.zeros(0, dtype=torch.float32, device=model.device)
    n_batch = 0
    for idx, batch in enumerate(dataloader):
        x, dx, tau = batch[0], batch[1], batch[2]
        dim_n = int(x.shape[-1] / 2)

        state_out = model(x, tau, ode)

        if ode:
            vel_pred = torch.split(state_out, split_size_or_sections=int(dim_n), dim=-1)[1]
            vel_true = torch.split(x, split_size_or_sections=int(dim_n), dim=-1)[1]
            sq_dist_state = torch.sum((vel_pred[:, 1:, :] - vel_true[:, 1:, :]) ** 2, dim=-1).reshape(-1, )
            state_pred_loss_mean = torch.mean(sq_dist_state)
        else:
            ddq_hat = torch.split(state_out, split_size_or_sections=int(dim_n), dim=-1)[1]
            ddq = torch.split(dx, split_size_or_sections=int(dim_n), dim=-1)[1]
            sq_dist_state = torch.sum((ddq_hat - ddq) ** 2, dim=-1)
            state_pred_loss_mean = torch.mean(sq_dist_state)

        loss = state_pred_loss_mean

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n_batch += 1
        epoch_loss += loss.detach().cpu().item()
        pred_err = torch.cat((pred_err, sq_dist_state))

    epoch_loss = epoch_loss / n_batch
    pred_mse = torch.mean(pred_err).detach().cpu().numpy()
    pred_var = torch.var(pred_err).detach().cpu().numpy()

    return [epoch_loss, pred_mse, pred_var]


def test_model_ode(dataloader, n, h, model):
    E_kin_gt = np.zeros(0, )
    E_kin_pred = np.zeros(0, )
    E_pot_gt = np.zeros(0, )
    E_pot_pred = np.zeros(0, )

    x_gt = np.zeros((0, h+1, 2*n))
    x_pred = np.zeros((0, h+1, 2*n))

    for idx, batch in enumerate(dataloader):
        x, dx, tau, E_kin, E_pot = batch[0], batch[1], batch[2], batch[4], batch[5]
        state_out = model(x, tau, do_ode=True)

        x_gt = np.append(x_gt, x.detach().cpu().numpy(), axis=0)
        x_pred = np.append(x_pred, state_out.detach().cpu().numpy(), axis=0)

        E_kin_hat = model.out_E_kin
        E_pot_hat = model.out_E_pot

        # append to numpy vectors
        E_kin_gt = np.append(E_kin_gt, E_kin.view(-1,).detach().cpu().numpy(), axis=0)
        E_kin_pred = np.append(E_kin_pred, E_kin_hat.view(-1,).detach().cpu().numpy(), axis=0)
        E_pot_gt = np.append(E_pot_gt, E_pot.view(-1,).detach().cpu().numpy(), axis=0)
        E_pot_pred = np.append(E_pot_pred, E_pot_hat.view(-1,).detach().cpu().numpy(), axis=0)


    plot_pos_vel_pred(x_gt, x_pred, model.delta_time_vec)


    return None


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_dof = 2
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/2d_pend/unact_small_dq0.pickle')
    save_model_params = False

    hyper = {'n_dof': n_dof,
             'n_width_V_eucl': 64,
             'n_depth_V_eucl': 2,
             'n_width_T_eucl': 64,
             'n_depth_T_eucl': 2,
             'exp_wrt_p': True,
             'activation': 'SoftPlus',
             'n_minibatch': 128,
             'learning_rate': 2e-4,
             'weight_decay': 1e-5,
             'max_epoch': 2500,
             'no_samples_train': 5000,
             'ode_solver': 'euler',
             'h_step_test': 'inf',
             'ode_timestep': -1.0,
             'n_delta_t': 10,
             'device': device}

    # fetch data for training & for testing
    training_data = DynDataset(n_dof, dataset_path, n_rec=[0, 39], phase='train',
                                      sampling_mode='random', n_samples=hyper['no_samples_train'],
                                      n_delta_t=hyper['n_delta_t'], device=device)
    testing_data = DynDatasetODE(n_dof, dataset_path, n_rec=[40, 42], phase='test', h_step_horizon=hyper['h_step_test'],
                              sampling_mode='uniform', n_delta_t=hyper['n_delta_t'], device=device)
    train_dataloader = DataLoader(training_data, batch_size=hyper['n_minibatch'], shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=hyper['n_minibatch'], shuffle=False)

    # initialize model
    lnn = LNN(**hyper).to(device)
    optim = RiemannianAdam(lnn.parameters(),
                                 lr=hyper['learning_rate'],
                                 weight_decay=hyper['weight_decay'],
                                 amsgrad=True)


    # training loop
    t0 = time.perf_counter()
    epoch_i = 0
    while epoch_i < hyper['max_epoch']:
        train_losses = train_epoch(train_dataloader, lnn, optim, ode=False, phase='train')

        epoch_i += 1
        if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
            print('current epoch number {0:05d}: '.format(epoch_i), end='', flush=True)
            print(f'train loss : {train_losses[0]:.4e} || ', end='', flush=True)
            print(f'mse: {train_losses[1]:.4e} || ', end='', flush=True)
            print(f'variance: {train_losses[2]:.4e} || ', end='', flush=True)
            print(f't since training start: {time.perf_counter() - t0:.4f}', flush=True)

    t_total = time.perf_counter() - t0
    print(f'finished training!! took {t_total:.2f} secs')
    if save_model_params:
        torch.save(lnn.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lnn_params.pth'))

    # evaluation of ode longterm prediction
    t_rec = next(iter(test_dataloader))[3]
    lnn.set_solver_times(t_rec)
    test_model_ode(test_dataloader, n_dof, testing_data.h_steps, lnn)

    plt.show()
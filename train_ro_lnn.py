import os, time, argparse, sys, pickle

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import platform

if platform.system() == 'Darwin':
    try:
        matplotlib.use('macosx')
    except ImportError:
        print("macosx backend not available; falling back to default.")


import torch
from torch.utils.data import DataLoader

from geoopt.optim import RiemannianAdam

from utils.DynData import DynDataset, DynDatasetODE
from models.reduced_LNN import RO_LNN


def train_epoch(dataloader, model, optimizer, phase='train'):
    epoch_loss = 0.0

    epoch_state_pred_err = torch.zeros(0, dtype=torch.float32, device=model.device)
    epoch_latent_pred_err = torch.zeros(0, dtype=torch.float32, device=model.device)
    epoch_ae_vel_pred_err = torch.zeros(0, dtype=torch.float32, device=model.device)
    epoch_ae_pos_pred_err = torch.zeros(0, dtype=torch.float32, device=model.device)
    n_batch = 0

    for idx, batch in enumerate(dataloader):
        x, dx, tau = batch[0], batch[1], batch[2]
        dim_n = int(x.shape[-1] / 2)
        q, dq = torch.split(x, split_size_or_sections=dim_n, dim=-1)

        if phase == 'test':
            with torch.no_grad():
                state_out, x_check, x_check_pred, x_hat = model(x, tau)
        else:
            state_out, x_check, x_check_pred, x_hat = model(x, tau)

        state_pred_err = torch.sum((state_out[:, 1:, dim_n:] - x[:, 1:, dim_n:]) ** 2, dim=-1).reshape(-1, )

        ae_pos_err = torch.sum((x_hat[:, :, :dim_n] - q)**2, dim=-1).reshape(-1, )
        ae_vel_err = torch.sum((x_hat[:, :, dim_n:] - dq) ** 2, dim=-1).reshape(-1, )
        latent_dim_n = int(x_check.shape[-1] / 2)
        latent_pred_err = torch.sum((x_check_pred[:, 1:, latent_dim_n:] - x_check[:, 1:, latent_dim_n:]) ** 2, dim=-1).reshape(-1, )

        loss = torch.mean(state_pred_err) + torch.mean(ae_pos_err) + torch.mean(ae_vel_err) + torch.mean(latent_pred_err)

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.autoencoder.update_parameters()

        n_batch += 1
        epoch_loss += loss.detach().cpu().item()
        epoch_state_pred_err = torch.cat((epoch_state_pred_err, state_pred_err))
        epoch_latent_pred_err = torch.cat((epoch_latent_pred_err, latent_pred_err))
        epoch_ae_vel_pred_err = torch.cat((epoch_ae_vel_pred_err, ae_vel_err))
        epoch_ae_pos_pred_err = torch.cat((epoch_ae_pos_pred_err, ae_pos_err))

    epoch_loss = epoch_loss / n_batch
    epoch_state_pred_mean = torch.mean(epoch_state_pred_err).detach().cpu().item()
    epoch_latent_pred_mean = torch.mean(epoch_latent_pred_err).detach().cpu().item()
    epoch_ae_vel_pred_mean = torch.mean(epoch_ae_vel_pred_err).detach().cpu().item()
    epoch_pos_pred_pred_mean = torch.mean(epoch_ae_pos_pred_err).detach().cpu().item()

    epoch_state_pred_var = torch.var(epoch_state_pred_err).detach().cpu().item()
    epoch_latent_pred_var = torch.var(epoch_latent_pred_err).detach().cpu().item()
    epoch_ae_vel_pred_var = torch.var(epoch_ae_vel_pred_err).detach().cpu().item()
    epoch_pos_pred_pred_var = torch.var(epoch_ae_pos_pred_err).detach().cpu().item()

    return [epoch_loss, epoch_state_pred_mean, epoch_state_pred_var, epoch_latent_pred_mean, epoch_latent_pred_var,
    epoch_pos_pred_pred_mean, epoch_pos_pred_pred_var,
    epoch_ae_vel_pred_mean, epoch_ae_vel_pred_var]

def test_model_ode(dataloader, n_dof, model, n_trajs, n_len_traj):
    q_pred_list, q_gt_list, dq_pred_list, dq_gt_list = [], [], [], []
    E_kin_gt_list, E_kin_pred_list, E_pot_gt_list, E_pot_pred_list = [], [], [], []
    for idx, batch in enumerate(dataloader):
        x, dx, tau, t, E_kin, E_pot = batch

        q, dq = torch.split(x, split_size_or_sections=int(n_dof), dim=-1)
        with torch.no_grad():
            state_out, x_check, x_check_pred, x_hat = model(x, tau)
        q_hat, dq_hat = torch.split(state_out, split_size_or_sections=int(n_dof), dim=-1)
        E_kin_hat = model.lagrangianNN.out_E_kin
        E_pot_hat = model.lagrangianNN.out_E_pot

        select_mask = torch.zeros((q.shape[0], q.shape[1]), dtype=torch.bool, device=model.device)
        select_mask[:, -1] = True
        if torch.any(t<1e-8):
            t0_idx = torch.where(t<1e-8)[0].item()
            select_mask[t0_idx] = True
        q_pred_list.append(q_hat[select_mask].detach().cpu().numpy())
        q_gt_list.append(q[select_mask].detach().cpu().numpy())
        dq_pred_list.append(dq_hat[select_mask].detach().cpu().numpy())
        dq_gt_list.append(dq[select_mask].detach().cpu().numpy())
        E_kin_pred_list.append(E_kin_hat[select_mask].detach().cpu().numpy())
        E_kin_gt_list.append(E_kin[select_mask].detach().cpu().numpy())
        E_pot_pred_list.append(E_pot_hat[select_mask].detach().cpu().numpy())
        E_pot_gt_list.append(E_pot[select_mask].detach().cpu().numpy())

    q_pred = np.concatenate(q_pred_list, axis=0)
    q_gt = np.concatenate(q_gt_list, axis=0)
    dq_pred = np.concatenate(dq_pred_list, axis=0)
    dq_gt = np.concatenate(dq_gt_list, axis=0)
    E_kin_pred = np.concatenate(E_kin_pred_list, axis=0)
    E_kin_gt = np.concatenate(E_kin_gt_list, axis=0)
    E_pot_pred = np.concatenate(E_pot_pred_list, axis=0)
    E_pot_gt = np.concatenate(E_pot_gt_list, axis=0)

    pos_mean = np.mean(np.linalg.norm(q_pred - q_gt, axis=1))
    pos_std = np.std(np.linalg.norm(q_pred - q_gt, axis=1))
    vel_mean = np.mean(np.linalg.norm(dq_pred - dq_gt, axis=1))
    vel_std = np.std(np.linalg.norm(dq_pred - dq_gt, axis=1))

    if n_dof > 6:
        skips = int(np.ceil(n_dof / 6))
        no_plots = 6
    else:
        skips = 1
        no_plots = n_dof

    fig = plt.figure(figsize=(12,7))
    fig.suptitle(f'RO-LNN: position and velocity prediction', fontsize=16, fontweight='bold')
    j = 0
    delta_t = model.lagrangianNN.delta_time_vec
    for i in range(0, n_dof, skips):
        ax_pos = fig.add_subplot(no_plots, 2, 2 * j + 1)
        ax_pos.plot(q_gt[:, i], color='k', label='ground truth')
        ax_pos.plot(q_pred[:, i], color='orange', label='RO-LNN')
        ax_pos.text(s=f'DoF {i+1}', rotation=90, x=-0.1, y=.5, horizontalalignment="center", verticalalignment="center", transform=ax_pos.transAxes)

        ax_vel = fig.add_subplot(no_plots, 2, 2 * (j + 1))
        ax_vel.plot(dq_gt[:, i], color='k', label='ground truth')
        ax_vel.plot(dq_pred[:, i], color='orange', label='RO-LNN')

        if j == 0:
            ax_pos.set_title(f'pos error mean:{pos_mean:.2e}, std:{pos_std:.2e}', pad=20)
            ax_vel.set_title(f'vel error mean:{vel_mean:.2e}, std:{vel_std:.2e}', pad=20)
            ax_vel.legend()

        ax_pos.set_xticks([])
        ax_vel.set_xticks([])
        tick_positions = np.array([0.0])
        tick_spacing = 4
        for n_dof in range(1, n_trajs + 1):
            new_traj_pos = n_dof * n_len_traj
            new_tick_pos = np.linspace(new_traj_pos - n_len_traj, new_traj_pos - 1, tick_spacing, dtype=int)
            tick_positions = np.append(tick_positions, new_tick_pos[1:])
            if n_dof < n_trajs:
                ax_pos.axvline(x=new_traj_pos, color='lightgrey')
                ax_vel.axvline(x=new_traj_pos, color='lightgrey')
            if j == 0:
                x_mid = 0.5 * (new_traj_pos + (n_dof - 1) * n_len_traj)
                ax_pos.text(x_mid, ax_pos.get_ylim()[1] * 1.15, f'trajectory {n_dof}', horizontalalignment='center', color='black', fontsize=9)
                ax_vel.text(x_mid, ax_vel.get_ylim()[1] * 1.15, f'trajectory {n_dof}', horizontalalignment='center', color='black', fontsize=9)
        tick_labels = np.tile(np.linspace(0, delta_t * n_len_traj, tick_spacing)[1:], n_trajs)
        tick_labels = np.insert(tick_labels, 0, 0.0)
        if j == 5:
            ax_pos.set_xticks(tick_positions)
            ax_pos.set_xticklabels([f"{x:.2f}" for x in tick_labels])
            ax_vel.set_xticks(tick_positions)
            ax_vel.set_xticklabels([f"{x:.2f}" for x in tick_labels])
            ax_pos.set_xlabel(f'time [s]')
            ax_vel.set_xlabel(f'time [s]')

        j += 1
    fig.tight_layout()


    return None

def test_model_ae(dataloader, n_dof, ndof_latent, model, n_trajs, n_len_traj):
    q_gt = np.zeros((0, n_dof))
    q_hat = np.zeros((0, n_dof))
    dq_gt = np.zeros((0, n_dof))
    dq_hat = np.zeros((0, n_dof))

    z = np.zeros((0, ndof_latent))
    dz = np.zeros((0, ndof_latent))

    z_tau = np.zeros((0, ndof_latent))

    for idx, batch in enumerate(dataloader):
        x, dx, tau = batch[0], batch[1], batch[2]
        q, dq = torch.split(x, split_size_or_sections=int(n_dof), dim=-1)
        _, ddq = torch.split(x, split_size_or_sections=int(n_dof), dim=-1)

        q_check, dq_check = model.autoencoder.encoder_vel(q, dx=dq)
        q_pred, dq_pred = model.autoencoder.decoder_vel(q_check, dz=dq_check)
        tau_check = model.autoencoder.reduce_act(q_check, tau)

        q_gt = np.append(q_gt, q.detach().cpu().numpy(), axis=0)
        q_hat = np.append(q_hat, q_pred.detach().cpu().numpy(), axis=0)
        dq_gt = np.append(dq_gt, dq.detach().cpu().numpy(), axis=0)
        dq_hat = np.append(dq_hat, dq_pred.detach().cpu().numpy(), axis=0)

        z = np.append(z, q_check.detach().cpu().numpy(), axis=0)
        dz = np.append(dz, dq_check.detach().cpu().numpy(), axis=0)
        z_tau = np.append(z_tau, tau_check.detach().cpu().numpy(), axis=0)

    pos_mean = np.mean(np.linalg.norm(q_hat - q_gt, axis=1))
    pos_std = np.std(np.linalg.norm(q_hat - q_gt, axis=1))
    vel_mean = np.mean(np.linalg.norm(dq_hat - dq_gt, axis=1))
    vel_std = np.std(np.linalg.norm(dq_hat - dq_gt, axis=1))

    if n_dof > 6:
        skips = int(np.ceil(n_dof / 6))
        no_plots = 6
    else:
        skips = 1
        no_plots = n_dof

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f'AE reconstruction on position and velocity', fontsize=16, fontweight='bold')
    delta_t = model.lagrangianNN.delta_time_vec
    j = 0
    for i in range(1, n_dof, skips):
        ax_pos = fig.add_subplot(no_plots, 2, 2 * j + 1)
        ax_pos.plot(q_gt[:, i], color='k', label='ground truth')
        ax_pos.plot(q_hat[:, i], color='orange', label='reconstruction')
        ax_pos.text(s=f'DoF {i+1}', rotation=90, x=-0.1, y=.5, horizontalalignment="center", verticalalignment="center", transform=ax_pos.transAxes)

        ax_vel = fig.add_subplot(no_plots, 2, 2 * j + 2)
        ax_vel.plot(dq_gt[:, i], color='k', label='ground truth')
        ax_vel.plot(dq_hat[:, i], color='orange', label='reconstruction')

        if j == 0:
            ax_pos.set_title(f'pos error mean:{pos_mean:.2e}, std:{pos_std:.2e}', pad=20)
            ax_vel.set_title(f'vel error mean:{vel_mean:.2e}, std:{vel_std:.2e}', pad=20)
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
            if j == 0:
                x_mid = 0.5 * (new_traj_pos + (n - 1) * n_len_traj)
                ax_pos.text(x_mid, ax_pos.get_ylim()[1] * 1.15, f'trajectory {n}', horizontalalignment='center', color='black', fontsize=9)
                ax_vel.text(x_mid, ax_vel.get_ylim()[1] * 1.15, f'trajectory {n}', horizontalalignment='center', color='black', fontsize=9)
        tick_labels = np.tile(np.linspace(0, delta_t * n_len_traj, tick_spacing)[1:], n_trajs)
        tick_labels = np.insert(tick_labels, 0, 0.0)
        if j == 5:
            ax_pos.set_xticks(tick_positions)
            ax_pos.set_xticklabels([f"{x:.2f}" for x in tick_labels])
            ax_vel.set_xticks(tick_positions)
            ax_vel.set_xticklabels([f"{x:.2f}" for x in tick_labels])
            ax_pos.set_xlabel(f'time [s]')
            ax_vel.set_xlabel(f'time [s]')
        j += 1

    fig.tight_layout()

    return None


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument('--train', action='store_true', help='flag set: re-train model')
    parser.add_argument('--save_params', action='store_true', help='flag set: train model without saving parameters')
    parser.add_argument('--save_model_to', type=str, default='pretrained/ro_lnn_cloth_v1', help="specify path where to save model parameters")

    parser.add_argument('--load_model_from', type=str, default='pretrained/ro_lnn_cloth', help="evaluate model from path")

    args = parser.parse_args()
    if not args.train:
        model_path = args.load_model_from
        # load configuration
        with open(os.path.join(model_path, 'summary.pickle'), 'rb') as f:
            summary = pickle.load(f)
        args = summary['args']
        hyper = summary['hyper']
        hyper['device'] = device

        # if interested, re-specify testing parameters here
        args.n_rec_test = [20, 22]
        hyper['h_step_test'] = 25

        # load model
        lrom = RO_LNN(**hyper).to(device)
        lrom.lagrangianNN.ae = lrom.autoencoder
        lrom.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location=device))
        lrom.autoencoder.update_parameters()

    else:
        args.n_dof = 600
        args.latent_dof = 10
        args.dataset_path = 'data/cloth/cloth_on_sphere.pickle'
        args.n_rec_train = [0, 19]
        args.n_rec_test = [20, 22]
        save_model_params = args.save_params
        hyper = {'layer_dims_AE': [args.n_dof, 128, 64, 32, args.latent_dof],
                 'n_width_V_eucl': 64,
                 'n_depth_V_eucl': 2,
                 'n_width_T_eucl': 64,
                 'n_depth_T_eucl': 2,
                 'exp_wrt_p': False,
                 'activation': 'SoftPlus',
                 'n_minibatch': 128,
                 'max_epoch': 2000,
                 'learning_rate': 2e-4,
                 'weight_decay': 2e-5,
                 'ode_solver': 'euler',
                 'ode_timestep': -1.0,
                 'h_step_train': 8,
                 'h_step_test': 25,
                 'no_samples_train': 2000,
                 'device': device}

        # fetch data for training & for testing
        training_data = DynDatasetODE(args.n_dof, args.dataset_path, n_rec=args.n_rec_train, phase='train',
                                      h_step_horizon=hyper['h_step_train'],
                                      sampling_mode='random', n_samples=hyper['no_samples_train'],
                                      device=device)
        train_dataloader = DataLoader(training_data, batch_size=hyper['n_minibatch'], shuffle=True)

        # initialize model
        lrom = RO_LNN(**hyper).to(device)
        optim_radam = RiemannianAdam([{'params': lrom.lagrangianNN.parameters(),
                                       'lr': hyper['learning_rate'],
                                       'weight_decay': hyper['weight_decay'],
                                       'amsgrad': True}, {'params': lrom.autoencoder.parameters(),
                                                          'lr': 500e-4,
                                                          'weight_decay': hyper['weight_decay'],
                                                          'amsgrad': True}])

        # training loop
        t_rec = next(iter(train_dataloader))[3]
        lrom.lagrangianNN.set_solver_times(t_rec)
        t0 = time.perf_counter()
        epoch_i = 0
        while epoch_i < hyper['max_epoch']:
            train_losses = train_epoch(train_dataloader, lrom, optim_radam,
                                                             phase='train')
            epoch_i += 1

            if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
                print('current epoch number {0:05d}: '.format(epoch_i), end='', flush=True)
                print(f'train loss: {train_losses[0]:.4e} || ', end='', flush=True)
                print(f'velocity MSE {train_losses[1]:.4e} || ', end='', flush=True)
                print(f'velocity var: {train_losses[2]:.4e} || ', end='', flush=True)
                print(f'lnn vel MSE {train_losses[3]:.4e} || ', end='', flush=True)
                print(f'lnn vel var: {train_losses[4]:.4e} || ', end='', flush=True)
                print(f'pos rec MSE {train_losses[5]:.4e} || ', end='', flush=True)
                print(f'pos rec var: {train_losses[6]:.4e} || ', end='', flush=True)
                print(f'vel rec MSE {train_losses[7]:.4e} || ', end='', flush=True)
                print(f'vel rec var: {train_losses[8]:.4e} || ', end='', flush=True)
                print(f't since training start: {time.perf_counter() - t0:.4f}', flush=True)

        t_total = time.perf_counter() - t0
        print(f'finished training!! took {t_total:.2f} secs')
        # save model
        if save_model_params:
            if not os.path.exists(args.save_model_to):
                os.makedirs(args.save_model_to)
            torch.save(lrom.state_dict(), os.path.join(args.save_model_to, 'model.pt'))
            result_summary = {'total_training_time': t_total,
                              'final_train_mse': train_losses[0],
                              'hyper': hyper,
                              'args': args}
            with open(os.path.join(args.save_model_to, 'summary.pickle'), 'wb') as f:
                pickle.dump(result_summary, f)

    testing_data_ode = DynDatasetODE(args.n_dof, args.dataset_path, n_rec=args.n_rec_test,
                                     h_step_horizon=hyper['h_step_test'], phase='test', sampling_mode='uniform',
                                     device=device)
    testing_data_acc = DynDataset(args.n_dof, args.dataset_path, n_rec=args.n_rec_test, phase='test',
                                  sampling_mode='uniform', device=device)
    test_dataloader_ode = DataLoader(testing_data_ode, batch_size=hyper['n_minibatch'], shuffle=False)
    test_dataloader_acc = DataLoader(testing_data_acc, batch_size=hyper['n_minibatch'], shuffle=False)

    # evaluate ode prediction
    t_rec = next(iter(test_dataloader_ode))[3]
    lrom.lagrangianNN.set_solver_times(t_rec)
    test_model_ode(test_dataloader_ode, args.n_dof, lrom, testing_data_acc.q.shape[0], testing_data_acc.q.shape[1])

    # evaluate AE
    test_model_ae(test_dataloader_acc, args.n_dof, args.latent_dof, lrom, testing_data_acc.q.shape[0],
                  testing_data_acc.q.shape[1])
    plt.show()

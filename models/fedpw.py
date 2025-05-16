import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel


class FedPW(FederatedModel):
    NAME = 'fedpw'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedPW, self).__init__(nets_list, args, transform)
        self.client_update = {}
        self.loss = {i: 0 for i in range(args.parti_num)}
        self.avg_loss_0 = 0
        self.avg_dot_0 = 0
        self.freq = {i: 1 / args.parti_num for i in range(args.parti_num)}
        self.norm_loss = {i: 1 / args.parti_num for i in range(args.parti_num)}
        self.delta_p = {i: 0 for i in range(args.parti_num)}
        self.p = {i: 0 for i in range(args.parti_num)}
        self.delta_p_dot = {i: 0 for i in range(args.parti_num)}
        self.p_dot = {i: 0 for i in range(args.parti_num)}
        self.total_drop_rate = {client_id: 0 for client_id in range(args.parti_num)}
        self.avg_drop_rate = args.avg_drop_rate

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(self.online_clients)

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
            net_params = self.nets_list[i].state_dict()
            global_params = self.global_net.state_dict()
            update_diff = {key: (global_params[key] - net_params[key]).float() for key in global_params}
            self.client_update[i] = update_diff
        freq = self.update()
        self.update_drop_rate()
        self.aggregate_nets_pw(freq)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        batch_loss = []
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                batch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()

        train_loss = sum(batch_loss) / len(batch_loss)
        self.loss[index] = train_loss

    def aggregate_nets_pw(self, freq=None):
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()
        if freq == None and self.args.averaging == 'weight':
            freq = {}
            online_clients_len = {}
            for i in online_clients:
                online_clients_len[i] = len(self.trainloaders[i].sampler)
            online_clients_all = sum(online_clients_len.values())
            for i in online_clients:
                freq[i] = online_clients_len[i] / online_clients_all
        elif freq == None:
            freq = {}
            online_num = len(online_clients)
            for i in online_clients:
                freq[i] = 1 / online_num
        self.freq = freq

        global_params_new = copy.deepcopy(global_w)

        for param_key in global_params_new:
            current_shape = self.client_update[0][param_key].shape
            sorted_online_clients = sorted(online_clients)
            grad_vec = torch.stack(
                [self.client_update[client_id][param_key].flatten() for client_id in sorted_online_clients])
            if grad_vec.dtype != torch.float32:
                grad_vec = grad_vec.to(torch.float32)
            if self.epoch_index == 0:
                freq_ = torch.tensor([freq[key] for key in sorted(freq.keys())],
                                     dtype=grad_vec.dtype).flatten().unsqueeze(1).to(self.device)
                g0 = torch.matmul(freq_.T, grad_vec).view(-1)
                global_params_new[param_key] = global_w[param_key] - g0.reshape(current_shape).to(self.device)
            else:
                g0 = self.drop(grad_vec, freq, param_key)
                global_params_new[param_key] = global_w[param_key] - g0.reshape(current_shape).to(self.device)

        print('\t\t'.join(f'{i}:{freq[i]:.3f}' for i in online_clients))
        self.global_net.load_state_dict(global_params_new)
        for i in online_clients:
            self.nets_list[i].load_state_dict(global_params_new)

    def drop(self, grad_vec, freq, param_key):
        dropped_values_mean = []
        for i in range(grad_vec.size(0)):
            drop_rate = self.total_drop_rate[i]
            num_elements_to_drop = int(grad_vec.size(1) * drop_rate)
            indices_to_drop = torch.topk(torch.abs(grad_vec[i]), num_elements_to_drop, largest=False).indices
            sampled_indices = indices_to_drop[::10]
            dropped_values_mean.append(grad_vec[i][sampled_indices].abs().mean().item())
            grad_vec[i][indices_to_drop] = 0
        dropped_values_mean = np.mean(dropped_values_mean)
        return self.consensus(grad_vec, freq, self.avg_drop_rate, dropped_values_mean, param_key)

    def consensus(self, grad_vec, freq, avg_drop_rate, dropped_values_mean, param_key):
        row_normalized_tensor = grad_vec / grad_vec.norm(p=2, dim=1, keepdim=True)
        column_normalized_tensor = row_normalized_tensor / row_normalized_tensor.norm(p=2, dim=0, keepdim=True)
        std_dev_per_column = torch.std(column_normalized_tensor, dim=0, unbiased=True)

        freq = torch.tensor([freq[key] for key in sorted(freq.keys())], dtype=grad_vec.dtype).flatten().unsqueeze(1).to(
            self.device)
        g0 = torch.matmul(freq.T, grad_vec).view(-1)
        if any(x in param_key for x in ['num_batches_tracked', 'running_mean', 'running_var']):
            return g0
        num_elements_to_drop = int(grad_vec.size(1) * avg_drop_rate)
        indices_to_rescale = torch.topk(torch.abs(std_dev_per_column), num_elements_to_drop, largest=False).indices
        sampled_indices = indices_to_rescale[::10]
        rescale_values_mean = g0[sampled_indices].abs().mean().item()
        if rescale_values_mean != 0 and dropped_values_mean / rescale_values_mean < 1:
            g0[indices_to_rescale] *= 1 + dropped_values_mean / rescale_values_mean
        return g0

    def update_drop_rate(self):
        num_tasks = self.args.parti_num
        total_loss = sum(self.loss.values())
        norm_loss = {i: self.loss[i] / total_loss for i in range(num_tasks)}
        total_norm_loss = sum(1 / norm_loss[i] for i in range(num_tasks))
        drop_rate = {i: min((1 / norm_loss[i]) / total_norm_loss * num_tasks * self.avg_drop_rate, 1) for i in range(num_tasks)}
        self.total_drop_rate = drop_rate

    def update(self):
        num_tasks = self.args.parti_num
        GG = torch.zeros(num_tasks, num_tasks)
        online_clients = self.online_clients
        para_key = [name for name, _ in self.nets_list[0].named_parameters()]

        for param_key in para_key:
            grad_vec = torch.stack(
                [self.client_update[client_id][param_key].flatten() for client_id in sorted(online_clients)]
            ).to(torch.float32)
            GG += torch.mm(grad_vec, grad_vec.t()).cpu()

        w_GG = torch.mv(GG, torch.ones(num_tasks))
        dot_avg = w_GG.mean()
        dot = {i: w_GG[i] / w_GG.sum() for i in range(num_tasks)}
        norm_loss = {i: self.loss[i] / sum(self.loss.values()) for i in range(num_tasks)}
        beta0 = self.args.beta
        loss_avg = sum(self.loss.values()) / num_tasks

        if self.epoch_index == 0:
            self.avg_loss_0 = loss_avg
            self.avg_dot_0 = dot_avg

        beta_loss = beta0 * loss_avg / self.avg_loss_0
        beta_dot = beta0 * dot_avg / self.avg_dot_0

        self.delta_p = {i: (1 - beta_loss) * self.delta_p[i] + beta_loss * norm_loss[i] for i in range(num_tasks)}
        self.p = {i: self.p[i] + self.delta_p[i] for i in range(num_tasks)}
        self.p = {i: self.p[i] / sum(self.p.values()) for i in range(num_tasks)}

        self.delta_p_dot = {i: (1 - beta_dot) * self.delta_p_dot[i] + beta_dot * dot[i] for i in range(num_tasks)}
        self.p_dot = {i: self.p_dot[i] + self.delta_p_dot[i] for i in range(num_tasks)}
        self.p_dot = {i: self.p_dot[i] / sum(self.p_dot.values()) for i in range(num_tasks)}

        loss_std = torch.std(torch.tensor(list(self.p.values())))
        dot_std = torch.std(torch.tensor(list(self.p_dot.values())))
        loss_ratio = loss_std / (loss_std + dot_std)
        dot_ratio = 1 - loss_ratio

        return {i: loss_ratio * self.p[i] + dot_ratio * self.p_dot[i] for i in range(num_tasks)}




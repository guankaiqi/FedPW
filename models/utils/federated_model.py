import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os
import copy
import math
from torch.nn.functional import cosine_similarity
class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)
        self.freq = None
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.online_clients_sequence = None
        self.trainloaders = None
        self.testloaders = None
        self.dataset_name_list = None 
        self.epoch_index = 0 
        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()
        self.accs = None
        self.client_update = {}
        self.total_drop_rate = {client_id: 0 for client_id in range(20)}
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)
        self.client_acc = {client_id: 0 for client_id in range(20)}
        self.net_to_device()



    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def ini(self):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        for net_id, net in enumerate(self.nets_list):
            self.prev_nets_list[net_id] = copy.deepcopy(net)




    def aggregate_nets(self, freq=None):
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
        first = True
        for net_id in online_clients:
            net = nets_list[net_id]
            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[net_id]

        print('\t\t'.join(f'{i}:{freq[i]:.3f}' for i in online_clients))

        self.global_net.load_state_dict(global_w)

        for i in online_clients:
            self.nets_list[i].load_state_dict(global_w)





from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

from torch.utils.data.sampler import Sampler, RandomSampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        # print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        # print(self.length)
        return self.length


class rgn(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        
        self.range = [3, 4, 5]

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        range_index = 0
        
        for pid in self.pids:
            # random.shuffle(self.range)
            # num_instances = self.range[range_index]
            num_instances = np.random.choice(self.range)
            # print(num_instances)
            # range_index += 1
            # if range_index >= len(self.range):
            #     range_index = 0
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < num_instances:
                # num_instances = len(idxs)
                idxs = np.random.choice(idxs, size=num_instances, replace=True)
            random.shuffle(idxs)
            # batch_idxs_dict[pid].append(idxs[:num_instances])
            # print(batch_idxs_dict[pid])
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
                    # num_instances = np.random.choice(self.range)
                    # print(num_instances)
                    # num_instances = self.range[range_index] #
                    num_instances = np.random.choice(self.range)
                    # range_index += 1
                    # if range_index >= len(self.range):
                    #     range_index = 0
            # for i in batch_idxs_dict[pid]:

            # print(batch_idxs_dict[pid])
            # print(idxs)
            if len(batch_idxs) > 3:
                batch_idxs_dict[pid].append(batch_idxs)
            # print(batch_idxs_dict[pid])

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) > self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                # print(len(batch_idxs))
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        # print(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        # print(self.length)
        return self.length


class SimRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.sim_index_dic = defaultdict(list)
        # self.include_sim = False
        for index, (_, pid, camid) in enumerate(self.data_source):
            
            if camid[1] != -1:
                self.sim_index_dic[pid].append(index)
            else:
                self.index_dic[pid].append(index)
            # self.include_sim = 
        self.pids = list(self.index_dic.keys())
        self.sim_pids = list(self.sim_index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % (self.num_instances // 2)

        # for pid in self.sim_pids:
        #     idxs = self.sim_index_dic[pid]
        #     num = len(idxs)
        #     if num < self.num_instances:
        #         num = self.num_instances
        #     self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        sim_batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        for pid in self.sim_pids:
            idxs = copy.deepcopy(self.sim_index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    sim_batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        avai_sim_pids = copy.deepcopy(self.sim_pids)
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch//2 and len(avai_sim_pids) >= (self.num_pids_per_batch - self.num_pids_per_batch//2):
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch//2)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
            selected_pids = random.sample(avai_sim_pids, self.num_pids_per_batch - self.num_pids_per_batch//2)
            for pid in selected_pids:
                batch_idxs = sim_batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(sim_batch_idxs_dict[pid]) == 0:
                    avai_sim_pids.remove(pid)
        # print(len(avai_sim_pids))
        # print(len(avai_pids))
        # print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        return 72720 #self.length


def build_train_sampler(data_source,
                        train_sampler,
                        train_batch_size,
                        num_instances,
                        **kwargs):
    """Build sampler for training
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - train_sampler (str): sampler name (default: RandomSampler).
    - train_batch_size (int): batch size during training.
    - num_instances (int): number of instances per identity in a batch (for RandomIdentitySampler).
    """

    if train_sampler == 'RandomIdentitySampler':
        print('random identity sampler')
        sampler = RandomIdentitySampler(data_source, train_batch_size, num_instances)
    elif train_sampler == 'sim':
        sampler = SimRandomIdentitySampler(data_source, train_batch_size, num_instances)
    elif train_sampler == 'rgn':
        print('rgn sampler')
        sampler = rgn(data_source, train_batch_size, num_instances)
    else:
        sampler = RandomSampler(data_source)

    return sampler

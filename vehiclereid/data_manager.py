from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, DoubleImageDataset
from .datasets import init_imgreid_dataset
from .transforms import build_transforms
from .samplers import build_train_sampler
from .utils.mean_and_std import  get_mean_and_std, calculate_mean_and_std
import xml.etree.ElementTree as ET
from collections import defaultdict
import math
import os
from glob import glob
from itertools import chain
from collections import OrderedDict

class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 include_sim=False,
                 root='datasets',
                 height=128,
                 width=256,
                 train_batch_size=64,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 illumination_aug=False,
                 random_erase=False,  # use random erasing for data augmentation
                 color_jitter=False,  # randomly change the brightness, contrast and saturation
                 color_aug=False,  # randomly alter the intensities of RGB channels
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug
        self.num_instances = num_instances
        self.illumination_aug = illumination_aug

        transform_train, transform_test = build_transforms(
            self.height, self.width, random_erase=self.random_erase, color_jitter=self.color_jitter,
            color_aug=self.color_aug, illumination_aug=self.illumination_aug
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict

    def return_sequential_veriwild(self, batch_size):
        path = 'datasets/veri-wild/vehicle_info.txt'
        test_path = 'datasets/veri-wild/test_3000.txt'
        test_query_path = 'datasets/veri-wild/test_3000_query.txt'
        items = defaultdict(list)
        test_ids = {}
        date_time_cars = OrderedDict()
        
        for i in range(5, 32):
            date_time_cars[f'2018-03-{str(i).zfill(2)}'] = [defaultdict(list) for _ in range(4)]
        for i in range(1, 7):
            date_time_cars[f'2018-04-{str(i).zfill(2)}'] = [defaultdict(list) for _ in range(4)]
        # {f'2018-03-{str(i).zfill(2)}': [defaultdict(list) for _ in range(4)] for i in range(5, 32)}
        # date_time_cars.update({f'2018-04-{str(i).zfill(2)}': [defaultdict(list) for _ in range(4)] for i in range(1, 7)})

        with open(test_path, 'r') as f:
            for l in f.readlines():
                l = l.strip()
                if l == '':
                    continue
                test_ids[l] = 0
        
        with open(test_query_path, 'r') as f:
            for l in f.readlines():
                l = l.strip()
                if l == '':
                    continue
                test_ids[l] = 0

        with open(path, 'r') as f:
            
            for idx, l in enumerate(f.readlines()):
                l = l.strip()
                if idx == 0 or l == '':
                    continue
                path, camid, time = l.split(';')[:3]
                # if path in test_ids:
                #     # print('pass')
                #     continue
                date = time.split(' ')
                date, time = date
                int_time = int(time.replace(':', ''))
                camid = int(camid)
                dataset = []
                date_time_cars[date][int(time.split(':')[0]) // 6][camid].append(
                    ['datasets/veri-wild/images/' + path + '.jpg', int(path.split('/')[0]), camid, int_time]
                )

                # time = time.replace(' ', '').replace(':', '').replace('-', '')
                # time = int(time)
                # camid = int(camid)
                # items[camid].append(['datasets/veri-wild/images/' + path + '.jpg', int(path.split('/')[0]), int(camid), time])
        dataloader_list = OrderedDict()
        total_length = 0
        for date, data in date_time_cars.items():
            tmp = []
            for items in data:
                for k in list(items.keys()):
                    items[k].sort(key= lambda x: x[-1])
                    # print(items[k])
                
                trains = []
                while True:
                    for k in list(items.keys()):
                        if len(items[k]) >= batch_size:
                            trains += items[k][:batch_size]
                            items[k] = items[k][batch_size:]
                        else:
                            trains += items[k]#[:batch_size]
                            del items[k]
                    
                    if len(items) == 0:
                        break

                trains = [g[:-1] for g in trains]
                print('length: ', len(trains))
                total_length += len(trains)
                tmp.append(DataLoader(
                    ImageDataset(trains, transform=self.transform_train),
                    batch_size=batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.use_gpu, drop_last=False
                ))
            dataloader_list[date] = tmp
        print(total_length)
        return dataloader_list

    def get_dataset(self, data):
        return DataLoader(
            ImageDataset(data, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

    def return_sequential_veri(self, batch_size):
        path='datasets/VeRi/train_label.xml'
        path_test='datasets/VeRi/test_label.xml'
        cp_ = [(5, 7), (4, 8), (3, 9), (11, 6, 10), (2, 1), (13, 12), (18, 17), (20, 19), (15, 14, 16)]
        cp = {}
        for idx, c in enumerate(cp_):
            for i in c:
                cp[i] = idx
        # cp = {for c in cp}
        xml = {}
        # print(path)
        root = ET.parse(path)
        # prefix = 333 if 'Sim' in path else 0
        # print(path)
        # print('prefix: ', prefix)
        for item in root.find('Items').findall('Item'):
            attr = item.attrib
            xml[attr['imageName']] = {
                'type': int(attr['typeID'])-1,
                'color': int(attr['colorID'])-1
            }

        # root = ET.parse(path_test)
        # xml_test = {}
        # for item in root.find('Items').findall('Item'):
        #     attr = item.attrib
        #     xml_test[attr['imageName']] = {
        #         'type': int(attr['typeID'])-1,
        #         'color': int(attr['colorID'])-1
        #     }
        times = {}
        # times = defaultdict(list)
        # splited, original
        xml = [[f.split('_')[:-1], f] for f in xml.keys()]
        # xml_test = [[f.split('_')[:-1], f] for f in xml_test.keys()]
        # car, cam, time
        # xml = [[int(x[0]), int(x[1][1:]), int(x[2]), x] for x in xml]
        # path, carid, camid, time
        xml = [[os.path.join(self.root, 'VeRi/image_train', ori), int(x[0]), int(x[1][1:]), int(x[2])] for x, ori in xml]
        test = glob(os.path.join(self.root, 'VeRi/image_test', '*'))
        test = [[os.path.basename(f).split('_')[:-1], os.path.basename(f)] for f in test]
        test = [[os.path.join(self.root, 'VeRi/image_test', ori), int(x[0]), int(x[1][1:]), int(x[2])] for x, ori in test]
        xml += test
        query = glob(os.path.join(self.root, 'VeRi/image_query', '*'))
        query = [[os.path.basename(f).split('_')[:-1], os.path.basename(f)] for f in query]
        query = [[os.path.join(self.root, 'VeRi/image_query', ori), int(x[0]), int(x[1][1:]), int(x[2])] for x, ori in query]
        xml += query
        # xml_test = [[os.path.join(self.root, 'VeRi/image_train', ori), int(x[0]), int(x[1][1:]), int(x[2])] for x, ori in xml_test]
        # times = set([x[-1] for x in xml])
        # for x in xml:
        #     times[math.floor(x[-1]/300)].append(x[:-1])
        
        for x in xml:
            if x[-1] not in times:
                times[x[-1]] = defaultdict(list)
            times[x[-1]][x[2]].append(x[:-1])

        # carid = {}
        # for x in xml:
        #     if x[-1] not in carid:
        #         carid[x[-1]] = defaultdict(list)
        #     carid[x[-1]][x[1]].append(x[:-1])

        
        # for x in xml:
        #     if math.floor(x[-1]/300) not in times:
        #         times[math.floor(x[-1]/300)] = defaultdict(list)
        #     times[math.floor(x[-1]/300)][math.floor(x[2]/10)].append(x[:-1])

        sorted_times = sorted(times.keys())
        train = []
        cars_per_camid = defaultdict(list)

        # carid: [cam cam cam]
        # moving_rule = defaultdict(list)
        # # for debug
        # for t in sorted_times:
        #     for cari, v in carid[t].items():
        #         if len(moving_rule[cari]) == 0:
        #             moving_rule[cari] += [i[2] for i in v]
        #         else:
        #             accepted = []
        #             for i in v:
        #                 if i[2] != moving_rule[cari][-1]:
        #                     accepted.append(i[2])
        #             moving_rule[cari] += accepted
        
        # for car, cams in moving_rule.items():
        #     print(f'car: {car}, cam: {cams}, set: {set(cams)}')
                # print(f'{t}, carid: {cari}, camid: {[i[2] for i in v]}')
        count = 0
        carid = 0
        freq = defaultdict(int)
        for t in sorted_times:
            for camid, v in times[t].items():
                cars_per_camid[camid] += v
                if len(cars_per_camid[camid]) >= batch_size:
                    train += cars_per_camid[camid][:batch_size]
                    tmp = defaultdict(int)
                    for i in cars_per_camid[camid][:batch_size]:
                        tmp[i[1]] += 1
                    for i, t in tmp.items():
                        freq[t] += 1
                        carid += t
                        count += 1
                    cars_per_camid[camid] = []#cars_per_camid[camid][batch_size//2:]
        print(carid/count)
        # total = sum([value for value in freq.values()])
        # print(freq)
        # for k, v in freq.items():
        #     print(k, ' ', v/total)
        # return
        # train = []

        # tmp = 0
        # c = 0
        # for t in sorted_times:
        #     t = times[t]
        #     # print([g for g in t.items()][0])

        #     tmp += sum([len(g) for g in t.values()])
        #     # print(tmp)
        #     c += len(t.values())
        #     # print(t.items())
        #     for path, pid, camid in chain(*(t.values())):
        #         # pid += self._num_train_pids
        #         # camid += self._num_train_cams
        #         train.append((path, pid, camid))
        # print(tmp/c)
        # return
        return DataLoader(
            ImageDataset(train, transform=self.transform_train),
            batch_size=batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )
        # total_id_per_sec = []
        # num_car_per_sec = []
        # total = 0
        # for t in sorted_times:
        #     v = times[t]
        #     vehicleid_count = defaultdict(int)
        #     # print(v)
        #     for n in v:
        #         vehicleid_count[n[0]] += 1
        #         total += 1
        #     print(t, ' ', vehicleid_count)
        #     mean = sum([f for f in vehicleid_count.values()]) / len(vehicleid_count)
        #     num_car_per_sec.append(sum([f for f in vehicleid_count.values()]))
        #     total_id_per_sec.append(mean)

    def return_unsupervised_dataset(self, name, batch_size, shuffle=True, drop_last=True, for_train=True):
        train = []
        dataset = init_imgreid_dataset(
            root=self.root, name=name, include_sim=False)

        for idx, (img_path, pid, camid) in enumerate(dataset.train):
            if isinstance(camid, list):
                if camid[1] != -1:
                    pid += self._num_train_pids
                camid[0] += self._num_train_cams
            else:
                pid += self._num_train_pids
                camid += self._num_train_cams
            train.append((img_path, idx, camid))

        return DataLoader(
            ImageDataset(train, transform=self.transform_train if for_train else self.transform_test),
            batch_size=batch_size, shuffle=shuffle, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=drop_last
        )

    def return_dataset(self, name):
        train = []
        dataset = init_imgreid_dataset(
            root=self.root, name=name, include_sim=False)
        
        for img_path, pid, camid in dataset.train:
            if isinstance(camid, list):
                if camid[1] != -1:
                    pid += self._num_train_pids
                camid[0] += self._num_train_cams
            else:
                pid += self._num_train_pids
                camid += self._num_train_cams
            train.append((img_path, pid, camid))
        
        train_sampler = build_train_sampler(
            train, 'RandomIdentitySampler',
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )
        # print(train)
        return DataLoader(
            ImageDataset(train, transform=self.transform_train),
            batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=True, sampler=train_sampler
        )

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Vehicle-ReID data manager
    """
    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)

        print('=> Initializing TRAIN (source) datasets')
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, include_sim=kwargs['include_sim'])

            for idx, (img_path, pid, camid) in enumerate(dataset.train):
                if isinstance(camid, list):
                    if camid[1] != -1:
                        pid += self._num_train_pids
                    camid[0] += self._num_train_cams
                else:
                    pid += self._num_train_pids
                    camid += self._num_train_cams
                train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        self.train_sampler = build_train_sampler(
            train, self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train), sampler=self.train_sampler,
            batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=True
        )
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]#calculate_mean_and_std(self.trainloader, len(train))
        # print('mean and std:', mean, std)

        print('=> Initializing TEST (target) datasets')
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in target_names}

        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, include_sim=kwargs['include_sim'])

            self.testloader_dict[name]['query'] = DataLoader(
                ImageDataset(dataset.query, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                ImageDataset(dataset.gallery, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train names      : {}'.format(self.source_names))
        print('  # train datasets : {}'.format(len(self.source_names)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(train)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test names       : {}'.format(self.target_names))
        print('  *****************************************')
        print('\n')

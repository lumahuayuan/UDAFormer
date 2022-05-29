import json
import os.path as osp
import os
from cv2 import exp
import mmcv
import numpy as np
import torch
import random
import math
from . import CityscapesDataset
from .builder import DATASETS

def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n 

    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def get_my_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n 

    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UDADataset(object):
    def __init__(self, source, target, cfg):
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE
        ogbs_cfg = cfg.get('ogbs')
        self.enable_ogbs = ogbs_cfg is not None

        if self.enable_ogbs:
            self.tau1 = ogbs_cfg['tau1']
            self.tau2 = ogbs_cfg['tau2'] 
            self.tau3 = ogbs_cfg['tau3'] 
            self.alpha = ogbs_cfg['alpha']
            self.beta = ogbs_cfg['beta']

            with open(osp.join(cfg['source']['data_root'],'sample_class_ratio.json'), 'r') as of:
                self.samples_with_class_ratio = json.load(of)
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i


    def ogbs_sampling(self, my_classprob, C, C_):
        C_1, C_2 = [], []
        for i in C_:
            for j in C:
                if (my_classprob[i,j] >= self.tau2) and (
                    abs(my_classprob[i,j]-my_classprob[j,i]) / (my_classprob[i,j]+my_classprob[j,i]) < self.tau3):
                    C_1.append(j)  
        C_1 = np.unique(C_1).tolist()

        for i in C_:
            for j in C:
                if (my_classprob[i,j] >= self.tau2) and (
                    abs(my_classprob[i,j]-my_classprob[j,i]) / (my_classprob[i,j]+my_classprob[j,i]) >= self.tau3):
                    C_2.append(j)
        C_2 = np.unique(C_2).tolist()

        sample_name, sample_prob = [], []
        for k,v in self.samples_with_class_ratio.items():
            sample_name.append(k)

            prob = 0
            for c in C:
                if v[str(c)] != 0:
                    prob += math.exp(v[str(c)])

            prob_ = 0
            for c_ in C_:
                if v[str(c_)] != 0:
                    prob_ += math.exp(v[str(c_)])

            prob_1 = 0
            for c_1 in C_1:
                if v[str(c_1)] != 0:
                    prob_1 += math.exp(v[str(c_1)])

            prob_2 = 0
            for c_2 in C_2:
                if v[str(c_2)] != 0:
                    prob_2 += math.exp(v[str(c_2)])

            prob = (prob_ + self.alpha * prob_1) / (prob + self.beta * prob_2)
            sample_prob.append(prob)
        
        sample_prob = (np.array(sample_prob) / np.sum(sample_prob)).tolist()
        f1 = np.random.choice(sample_name, p=sample_prob)
        return f1


    def get_ogbs_sample(self):
        json_path = './mylog'
        files = os.listdir(json_path)
        f_new = max([int(f.split('_')[-1].split('.')[0]) for f in files])
        with open(osp.join(json_path, 'matrix_{0}.json'.format(f_new)), 'r') as of:
            myjson = json.load(of)
        
        my_classprob = torch.tensor(np.array([v for k,v in myjson.items()]))

        C, C_ = [i for i in range(my_classprob.shape[0])], []
        for i in C:
            if my_classprob[i,i] < self.tau1:
                C_.append(i)

        if len(C_) == 0:
            f1 = random.sample(self.samples_with_class_ratio.keys(), 1)[0]
            if not isinstance(f1, str):
                a=1
        else:
            f1 = self.ogbs_sampling(my_classprob, C, C_)

        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]  
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]

        return {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }


    def __getitem__(self, idx):
        if self.enable_ogbs:
            return self.get_ogbs_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
            return {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img']
            }


    def __len__(self):
        return len(self.source) * len(self.target)

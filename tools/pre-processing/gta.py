import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('--gta_path', default='data/gta', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_sample_class_ratio(out_dir, sample_class_stats):
    sample_class_ratio = {}
    for data in sample_class_stats:
        name = data.pop('file').split('\\')[-1]
        sample_class_ratio[name] = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,
                                    '10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,'17':0,'18':0}
        n_sum = 0
        for c, n in data.items():
            n_sum += n
        for c, n in data.items():
            sample_class_ratio[name][c] = n / n_sum

    with open(osp.join(out_dir, 'sample_class_ratio.json'), 'w') as of:
        json.dump(sample_class_ratio, of, indent=2)

def main():
    args = parse_args()
    gta_path = args.gta_path
    out_dir = args.out_dir if args.out_dir else gta_path
    mmcv.mkdir_or_exist(out_dir)

    with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)

    save_sample_class_ratio(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()

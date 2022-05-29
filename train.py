import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import torch
from mmcv import Config, get_git_hash
from tools import train


def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        type=int,
        default=None,
        help='Experiment id as defined in experiment.py',
    )
    parser.add_argument(
        '--config',
        default='configs/udaformer/udaformer_ts_er_ogbs_mitb5.py',
        help='Path to config file',
    )
    parser.add_argument(
        '--machine', type=str, choices=['local'], default='local')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'
    cfgs, config_files = [], []

    if args.config is not None:
        cfg = Config.fromfile(args.config)
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash()
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)

        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    if args.machine == 'local':
        for i, cfg in enumerate(cfgs):
            print('Run job {}'.format(cfg['name']))
            train.main([config_files[i]])
            torch.cuda.empty_cache()
    else:
        raise NotImplementedError(args.machine)

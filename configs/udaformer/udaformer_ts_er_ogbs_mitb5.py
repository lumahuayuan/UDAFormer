_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/udaformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0
uda = dict(
    alpha=0.999,
    # 'building','pole','traffic_light','traffic_sign','person','rider','car','truck','bus','train','motorcycle','bicycle'
    structured_feature_dist_classes=[2, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # 'road','sidewalk','walk','fence','vegetation','terrain', 'sky'
    unstructured_feature_dist_classes=[0, 1, 3, 4, 8, 9, 10],
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    teacher_student_er_lambda=0.005,
    # teacher_student_er_lambda=-1,
    encoder_regular_dist_classes=None
    )


data = dict(
    train=dict(
        ogbs=dict(tau1=0.7, tau2=0.1, tau3=0.6, alpha=1.0, beta=1.0)))

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=-1)
evaluation = dict(interval=2000, metric='mIoU')

name = 'udaformer_ts_er_ogbs_mitb5'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'udaformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'udaformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

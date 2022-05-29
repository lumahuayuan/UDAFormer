import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from torch.autograd import Variable

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform, strong_transform_nomix)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):
    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.er_lambda = cfg['teacher_student_er_lambda']
        self.structured_classes = cfg['structured_feature_dist_classes']
        self.unstructured_classes = cfg['unstructured_feature_dist_classes']
        self.encoder_classes = cfg['encoder_regular_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_er = self.er_lambda >= 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability'] 
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'
        self.debug_fdist_mask = None
        self.debug_gt_rescale = None
        self.class_probs = {}
        self.ema_model = build_segmentor(deepcopy(cfg['model']))

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()

        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(), self.get_model().parameters()):
            if not param.data.shape:
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        pw_feat_dists = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dists = pw_feat_dists[mask.squeeze(1)]    

        pw_feat_dist = torch.mean(pw_feat_dists)
        if torch.any(torch.isnan(pw_feat_dist)):
            pw_feat_dist = torch.tensor([0.0], requires_grad=True)
        
        assert not torch.any(torch.isnan(pw_feat_dist))
        return pw_feat_dist

    def calc_encoder_regular(self, img, gt, feat_strong):
        assert self.enable_er
        with torch.no_grad():
            self.get_ema_model().eval()
            feat_week = self.get_ema_model().extract_feat(img) 
            feat_week = [f.detach() for f in feat_week]

        lay = -1
        if self.encoder_classes is not None:
            fdclasses = torch.tensor(self.encoder_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat_week[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1) 

            feat_dist = self.masked_feat_dist(feat_week[lay], feat_strong[lay], fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat_week[lay], feat_strong[lay])
        
        feat_dist = self.er_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_encoder_regular': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log


    def calc_structure_kl(self, student, teacher, pseudo_label):
        pseudo_label = pseudo_label.unsqueeze(1)
        if self.structured_classes is not None:
            fdclasses = torch.tensor(self.structured_classes, device=pseudo_label.device)
            scale_factor = pseudo_label.shape[-1] // student.shape[-1]

            pseudo_label_rescaled = downscale_label_ratio(pseudo_label, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            
            kl_mask = torch.any(pseudo_label_rescaled[..., None] == fdclasses, -1) 
            kl_mask = kl_mask.repeat(1, student.shape[1], 1, 1)
            kl_loss = F.kl_div(student, teacher, reduction='none') 
            kl_loss = 1 * kl_loss[kl_mask].sum() / (student.shape[0]*student.shape[2]*student.shape[3])

        return kl_loss


    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_p': self.color_jitter_p,
            'color_jitter_s': self.color_jitter_s,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),
            'std': stds[0].unsqueeze(0)
        }

        if self.enable_er:
            source_img_strong, source_lbl_strong = [None] * batch_size, [None] * batch_size
            for i in range(batch_size):
                source_img_strong[i], source_lbl_strong[i] = strong_transform_nomix(
                    strong_parameters,
                    data=img[i].clone(),
                    target=gt_semantic_seg[i].clone())

            source_img_strong = torch.cat(source_img_strong)
            source_lbl_strong = torch.cat(source_lbl_strong)
            img_week_strong = [img, source_img_strong]
            source_losses_week, _ = self.get_model().forward_train(
                img_week_strong, img_metas, gt_semantic_seg, return_feat=True)
            source_feat_strong = source_losses_week.pop('features_strong')
        
        else:
            source_losses_week, _ = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, return_feat=True)

        source_loss, source_log = self._parse_losses(source_losses_week)
        log_vars.update(source_log)
        source_loss.backward(retain_graph=self.enable_er)

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits, _ = self.get_ema_model().encode_decode(target_img, target_img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))

        pseudo_weight = torch.sum(ps_large_p).item() / ps_size 
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)
        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0 
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0 
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        if self.enable_er:
            encoder_regular_loss, regular_log = self.calc_encoder_regular(img, gt_semantic_seg, source_feat_strong)
            encoder_regular_loss.backward(retain_graph=True)
            log_vars.update(add_prefix(regular_log, 'encoder regular'))

        target_img_strong, target_lbl_strong = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            target_img_strong[i], target_lbl_strong[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            
        target_img_strong = torch.cat(target_img_strong)
        target_lbl_strong = torch.cat(target_lbl_strong)

        target_losses, student_logits = self.get_model().forward_train(
            target_img_strong, img_metas, target_lbl_strong, pseudo_weight, return_feat=True)
        
        target_losses.pop('features_week')
        target_losses = add_prefix(target_losses, 'target')
        target_loss, target_log = self._parse_losses(target_losses) 
        log_vars.update(target_log)
        target_loss.backward()
        self.local_iter += 1

        return log_vars

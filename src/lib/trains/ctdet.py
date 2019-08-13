from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

from torch.nn import functional as F
import matplotlib.pyplot as plt



def mask_show(croped_mask, mask):
    croped_mask = croped_mask.data.cpu().numpy()
    mask = mask.data.cpu().numpy()
    mask *= 255

    plt.subplot(1, 2, 1)
    plt.imshow(croped_mask)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h


def sanitize_coordinates(_x1, _x2, img_size: int, padding: int=0, cast: bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    # _x1 = _x1 * img_size
    # _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)

    return x1, x2


def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    def lincomb_mask_loss(self, mask_data, proto_data,
                          reg_mask_gt, masks_gt, gt_bbox_lbl, interpolation_mode='bilinear'):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        batch_size = mask_data.size(0)

        # process_gt_bboxes = cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop

        # if cfg.mask_proto_remove_empty_masks:
        #     # Make sure to store a copy of this because we edit it to get rid of all-zero masks
        #     pos = pos.clone()

        loss_m = 0
        # loss_d = 0  # Coefficient diversity loss
        total_num_pos = 0

        for idx in range(batch_size):
            if masks_gt[idx].sum() == 0:
                continue

            with torch.no_grad():
                downsampled_masks = F.interpolate(masks_gt[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                # if cfg.mask_proto_binarize_downsampled_gt:
                downsampled_masks = downsampled_masks.gt(0.5).float()
                mask_t = downsampled_masks

            # cur_pos = pos[idx]
            # pos_idx_t = idx_t[idx, cur_pos]
            # pos_idx_t = (reg_mask_gt[idx].data.cpu().numpy() == 1)

            # if process_gt_bboxes:
            #     # Note: this is in point-form
            #     pos_gt_box_t = gt_box_t[idx, cur_pos]
            #     pos_gt_box_t = gt_box_t[idx, cur_pos]
            # ct = centers_gt[idx][pos_idx_t, :]
            # wh = wh_gt[idx][pos_idx_t, :]
            pos_gt_box_t = gt_bbox_lbl[idx][:, :4].data

            # outputs of ProtoNet
            proto_masks = proto_data[idx]

            # mask coefficients
            bboxes = gt_bbox_lbl[idx][:, :4].data.cpu().numpy()
            ct = np.array([(bboxes[:, 0] + bboxes[:, 2]) / 2,
                           (bboxes[:, 1] + bboxes[:, 3]) / 2], dtype=np.float32)
            centers = ct.astype(np.int32)
            proto_coef = mask_data[idx, :, centers[0], centers[1]]

            # Test centers of gt bboxes
            # centers = centers_gt[idx]
            # centers = centers[pos_idx_t, :]
            # centers = np.stack(centers)
            # proto_coef_2 = mask_data[idx, :, centers[:, 0], centers[:, 1]]

            # if pos_idx_t.size(0) == 0:
            #     continue

            # proto_masks = proto_data[idx]
            # proto_coef = mask_data[idx, cur_pos, :]

            # if cfg.mask_proto_coeff_diversity_loss:
            #     if inst_data is not None:
            #         div_coeffs = inst_data[idx, cur_pos, :]
            #     else:
            #         div_coeffs = proto_coef
            #
            #     loss_d += self.coeff_diversity_loss(div_coeffs, pos_idx_t)

            # If we have over the allowed number of masks, select a random sample
            # old_num_pos = proto_coef.size(0)
            # if old_num_pos > cfg.masks_to_train:
            #     perm = torch.randperm(proto_coef.size(0))
            #     select = perm[:cfg.masks_to_train]
            #
            #     proto_coef = proto_coef[select, :]
            #     pos_idx_t = pos_idx_t[select]
            #
            #     if process_gt_bboxes:
            #         pos_gt_box_t = pos_gt_box_t[select, :]

            # num_pos = proto_coef.size(0)
            # mask_t = downsampled_masks[:, :, pos_idx_t]

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef
            pred_masks = torch.sigmoid(pred_masks)

            pred_masks = crop(pred_masks, pos_gt_box_t)
            # pred_masks = crop(downsampled_masks, pos_gt_box_t)
            # mask_show(pred_masks[:, :, 0], downsampled_masks[:, :, 0])

            pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')

            weight = mask_h * mask_w
            pos_get_csize = center_size(pos_gt_box_t)
            gt_box_width = pos_get_csize[:, 2] # * mask_w
            gt_box_height = pos_get_csize[:, 3] # * mask_h
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight



            # if cfg.mask_proto_double_loss:
            #     if cfg.mask_proto_mask_activation == activation_func.sigmoid:
            #         pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='sum')
            #     else:
            #         pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='sum')
            #
            #     loss_m += cfg.mask_proto_double_loss_alpha * pre_loss
            #
            # if cfg.mask_proto_crop:
            #     pred_masks = crop(pred_masks, pos_gt_box_t)
            #
            # if cfg.mask_proto_mask_activation == activation_func.sigmoid:
            #     pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')
            # else:
            #     pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')
            #
            # if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
            #     gt_area = torch.sum(mask_t, dim=(0, 1), keepdim=True)
            #     pre_loss = pre_loss / (torch.sqrt(gt_area) + 0.0001)
            #
            # if cfg.mask_proto_reweight_mask_loss:
            #     pre_loss = pre_loss * mask_reweighting[:, :, pos_idx_t]
            #
            # if cfg.mask_proto_normalize_emulate_roi_pooling:
            #     weight = mask_h * mask_w if cfg.mask_proto_crop else 1
            #     pos_get_csize = center_size(pos_gt_box_t)
            #     gt_box_width = pos_get_csize[:, 2] * mask_w
            #     gt_box_height = pos_get_csize[:, 3] * mask_h
            #     pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight

            # # If the number of masks were limited scale the loss accordingly
            # if old_num_pos > num_pos:
            #     pre_loss *= old_num_pos / num_pos

            loss_m += torch.sum(pre_loss)

            # Divide loss_m by the number of positives (number of gt bounding boxes).
            total_num_pos += bboxes.shape[0]

        # TODO: Move to config file
        mask_alpha = 0.4 / 256 * 140 * 140
        losses = loss_m * mask_alpha / mask_h / mask_w / total_num_pos

        # if cfg.mask_proto_coeff_diversity_loss:
        #     losses['D'] = loss_d

        return losses

    def semantic_segmentation_loss(self, segment_data, mask_t, bbox_lbl, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0

        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = bbox_lbl[idx][:, -1].data.long()

            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()

                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]],
                                                                downsampled_masks[obj_idx])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')

        # TODO: Move to config file
        semantic_segmentation_alpha = 1
        return loss_s / mask_h / mask_w * semantic_segmentation_alpha / batch_size  # cfg.semantic_segmentation_alpha

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, lincomb_mask_loss, segm_loss = 0, 0, 0, 0, 0
        # hm_loss, wh_loss, off_loss, segm_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]

            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                batch['dense_wh'] * batch['dense_wh_mask']) /
                                mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.use_semantic_segmentation_loss:
                segm_loss += self.semantic_segmentation_loss(output['segm'],
                                                             batch['masks'],
                                                             batch['gt_bbox_lbl']) / opt.num_stacks
            lincomb_mask_loss += self.lincomb_mask_loss(output['masks'], output['proto'],
                                                        batch['reg_mask'], batch['masks'],
                                                        batch['gt_bbox_lbl']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + \
               opt.lincomb_mask_weight * lincomb_mask_loss + opt.segm_weight * segm_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss,
                      'lincomb_mask_loss': lincomb_mask_loss, 'segm_loss': segm_loss}
        # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
        #        opt.off_weight * off_loss +  opt.segm_weight * segm_loss
        # loss_stats = {'loss': loss, 'hm_loss': hm_loss,
        #               'wh_loss': wh_loss, 'off_loss': off_loss, 'segm_loss': segm_loss}
        return loss, loss_stats


class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'lincomb_mask_loss', 'segm_loss']
        # loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'segm_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
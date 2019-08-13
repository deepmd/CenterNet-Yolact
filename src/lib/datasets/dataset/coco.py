from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class COCO(data.Dataset):
    num_classes = 80
    # num_classes = 8
    default_resolution = [544, 544]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    rgb = False

    # Pelee mean, std, rgb
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    # rgb  = True

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, 'images/{}2017'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(split)
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_extreme_{}2017.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'instances_{}2017.json').format(split)
        self.max_objs = 128
        self.class_name = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

        # self.class_name = [
        #   '__background__', 'airplane', 'train', 'stop sign', 'parking meter',
        #   'cat', 'umbrella', 'handbag', 'spoon']
        # self._valid_ids = [5, 7, 13, 14, 17, 28, 31, 50]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        # self.images = []
        # for cat in self.coco.getCatIds(catNms=self.class_name):
        #     self.images.extend(self.coco.getImgIds(catIds=cat))
        # CenterNet code
        # self.images = self.coco.getImgIds()

        # Yolact code
        self.images = list(self.coco.imgToAnns.keys())
        if len(self.images) == 0:
            self.images = list(self.coco.imgs.keys())
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def get_detection_eval_metrics(self, coco_eval):
        det_eval = {}
        total_precisions = {}
        total_recalls = {}
        avg_precisions = {}

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        # precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        # ap_default = np.mean(precision[precision > -1])
        # out_str += '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~ \n'.format(IoU_lo_thresh, IoU_hi_thresh)
        # out_str += 'mAP: {:.1f}'.format(100 * ap_default)
        for cls_ind, cls in enumerate(self.class_name):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            total_recalls[cls] = coco_eval.eval['recall'][ind_lo:(ind_hi + 1), cls_ind - 1, 0, 2]
            last_recalls = (total_recalls[cls] * 100).astype(np.int)
            total_precisions[cls] = np.array(
                [coco_eval.eval['precision'][iou_ind, rec_ind, cls_ind - 1, 0, 2]
                 for iou_ind, rec_ind in zip(range(ind_lo, (ind_hi + 1)), last_recalls)])
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            avg_precisions[cls] = np.mean(precision, axis=-1)
            map = np.mean(precision[precision > -1])
            det_eval[cls] = map
            # out_str += '{}: {:.1f} \n'.format(cls, 100 * ap)

        print('\n~~~~ Summary metrics ~~~~\n')
        coco_eval.summarize()
        return det_eval, avg_precisions, total_precisions, total_recalls

    def format_output(self, stats):
        out = {}
        out['AP IoU=0.50:0.95 area=all maxDets=100'] = stats[0]
        out['AP IoU=0.50      area=all maxDets=100'] = stats[1]
        out['AP IoU=0.75      area=all maxDets=100'] = stats[2]
        out['AP IoU=0.50:0.95 area=small maxDets=100 '] = stats[3]
        out['AP IoU=0.50:0.95 area=medium maxDets=100'] = stats[4]
        out['AP IoU=0.50:0.95 area=large  maxDets=100'] = stats[5]
        out['AR IoU=0.50:0.95 area=all maxDets=1'] = stats[6]
        out['AR IoU=0.50:0.95 area=all maxDets=10'] = stats[7]
        out['AR IoU=0.50:0.95 area=all maxDets=100'] = stats[8]
        out['AR IoU=0.50:0.95 area=small maxDets=100'] = stats[9]
        out['AR IoU=0.50:0.95 area=medium maxDets=100'] = stats[10]
        out['AR IoU=0.50:0.95 area= large maxDets=100'] = stats[11]
        return out

    def run_eval(self, results, save_dir, precision_recall=False):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        det_eval, avg_precisions, total_precisions, total_recalls = self.get_detection_eval_metrics(coco_eval)
        stats = self.format_output(coco_eval.stats)
        if precision_recall:
            return avg_precisions, total_precisions, total_recalls
        else:
            return stats, det_eval

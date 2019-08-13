from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import pandas as pd
import time
from progress.bar import Bar
import torch

from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)


def detect(opt, dataset):
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = 'detection: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()
  bar.finish()
  return results


def get_metrics(detections, dataset, thresh, exp_id):
  filtered_detection = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(exp_id), max=num_iters)
  for ind, (img_id, results) in enumerate(detections.items()):
    filtered_detection[img_id] = \
      {cls_id: boxes[boxes[:, 4] > thresh] for cls_id, boxes in results.items()}
    Bar.suffix = 'thresh {:.2f}: [{}/{}]|Tot: {} |ETA: {} '.format(
      thresh, ind, num_iters, bar.elapsed_td, bar.eta_td)
    bar.next()
  bar.finish()
  cls_APs, cls_precisions, cls_recalls = dataset.run_eval(filtered_detection, opt.save_dir, precision_recall=True)
  return cls_APs, cls_precisions, cls_recalls


if __name__ == '__main__':
  start = 0.01
  end = 0.95
  step = 0.01
  at05 = True

  opt = opts().parse()
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  output_path = os.path.join(opt.save_dir, 'cn_thresholds{}_{:.2f}_{:.2f}_{:.2f}.xlsx'
                             .format(('','@0.5')[at05], start, end, step))

  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)

  detections = detect(opt, dataset)

  num_metrics = 4
  thresh_range = np.arange(start, end, step)
  classes_best_thresh = [(0, 0)] * (dataset.num_classes*num_metrics)
  columns = ['class', 'best_thresh'] + list(thresh_range)
  df = pd.DataFrame(columns=columns)
  for cls in dataset.class_name[1:]:
    df = df.append([pd.Series([cls, 0], index=['class', 'best_thresh'])]*num_metrics, ignore_index=True)
  df.to_excel(output_path)

  for thresh in thresh_range:
    cls_APs, cls_precisions, cls_recalls = get_metrics(detections, dataset, thresh, opt.exp_id)
    for cls in cls_APs:
      cls_ind = dataset.class_name.index(cls) - 1
      ind = cls_ind * num_metrics
      if at05:  # iou@0.5
        mAP = cls_APs[cls][0]
        precision = cls_precisions[cls][0]
        recall = cls_recalls[cls][0]
      else:  # iou@0.5-0.95
        mAP = np.mean(cls_APs[cls])
        precision = np.mean(cls_precisions[cls])
        recall = np.mean(cls_recalls[cls])
      f1score = 2*precision*recall / (precision+recall)
      if mAP > classes_best_thresh[ind][0]:
        classes_best_thresh[ind] = (mAP, thresh)
      if precision > classes_best_thresh[ind + 1][0]:
        classes_best_thresh[ind + 1] = (precision, thresh)
      if recall > classes_best_thresh[ind + 2][0]:
        classes_best_thresh[ind + 2] = (recall, thresh)
      if f1score > classes_best_thresh[ind + 3][0]:
        classes_best_thresh[ind + 3] = (f1score, thresh)
      df.loc[[ind], [thresh]] = mAP
      df.loc[[ind + 1], [thresh]] = precision
      df.loc[[ind + 2], [thresh]] = recall
      df.loc[[ind + 3], [thresh]] = f1score
    df.to_excel(output_path)
    print()


  for i, (_, best_thresh) in enumerate(classes_best_thresh):
    df.loc[[i], ['best_thresh']] = best_thresh
  df = df.set_index(['class', 'best_thresh'])
  df.to_excel(output_path)

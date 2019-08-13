from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import multiprocessing

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt, benchmark=False):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  if not benchmark:
    opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    total_time = 0.0
    count = 0.0
    for i in range(50 if benchmark else 1):
        for (image_name) in image_names:
          ret = detector.run(image_name)
          time_str = ''
          total_time += ret['pre'] + ret['net'] + ret['dec'] + ret['post']
          count += 1.0
          for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          print(time_str)
    print('average total time: {:.3f}s'.format(total_time/count))

if __name__ == '__main__':
  dataset = None
  dataset = {'default_resolution': [512, 512], 'num_classes': 96,
             'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
             'dataset': 'coco+oi'}
  opt = opts().init(dataset=dataset)
  # p1 = multiprocessing.Process(name='p1', target=demo, args=[opt, True])
  # p2 = multiprocessing.Process(name='p2', target=demo, args=[opt, True])
  # p1.start()
  # p2.start()
  # demo(opt, True)
  demo(opt)
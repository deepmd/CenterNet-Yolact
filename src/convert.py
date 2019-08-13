import _init_paths

from models.model import create_model, load_model
from opts import opts

import torch
from torch.autograd import Variable
from pytorch2caffe import pytorch_to_caffe as p2c

name = 'CenterNet_shelf_res18_510_80'
img_dim = 510  # (384, 512)
num_classes = 80  # VOC=21 COCO=81
dataset = {'default_resolution': [img_dim, img_dim], 'num_classes': num_classes,
           'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
           'dataset': 'coco+oi'}

opt = opts().init(dataset=dataset)
net = create_model(opt.arch, opt.heads, opt.head_conv)
# net = load_model(net, opt.load_model)
net.eval()

input = Variable(torch.ones([1, 3, img_dim, img_dim]))
p2c.trans_net(net, input, name, True)
p2c.save_prototxt('../caffe_models/{}.prototxt'.format(name))
p2c.save_caffemodel('../caffe_models/{}.caffemodel'.format(name))

print('Pytorch model "{}" conversion completed.'.format(name))

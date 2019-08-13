from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from trains.evaluator import Evaluator


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    input = []
    hm = []
    reg_mask = []
    ind = []
    wh = []
    reg = []
    masks = []
    # labels = []
    # num_crowds = []
    # centers = []
    gt_bbox_lbl = []
    for sample in batch:
        if len(sample['gt_bbox_lbl']) > 0:
            input.append(torch.FloatTensor(sample['input']))
            hm.append(torch.FloatTensor(sample['hm']))
            reg_mask.append(torch.ByteTensor(sample['reg_mask']))
            ind.append(torch.LongTensor(sample['ind']))
            wh.append(torch.FloatTensor(sample['wh']))
            reg.append(torch.FloatTensor(sample['reg']))
            masks.append(torch.FloatTensor(sample['masks']))
            gt_bbox_lbl.append(torch.FloatTensor(sample['gt_bbox_lbl']))
            # labels.append(torch.ByteTensor(sample['labels']))
            # num_crowds.append(torch.ByteTensor(sample['num_crowds']))
            # num_crowds.append(sample['num_crowds'])
            # centers.append(sample['centers'])

    return {'input': torch.stack(input, 0),
            'hm': torch.stack(hm, 0),
            'reg_mask': torch.stack(reg_mask, 0),
            'ind': torch.stack(ind, 0),
            'wh': torch.stack(wh, 0),
            'reg': torch.stack(reg, 0),
            'masks': masks,
            # 'labels': labels,
            # 'num_crowds': num_crowds,
            # 'centers': centers,
            'gt_bbox_lbl': gt_bbox_lbl
            }


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.freeze)

    # count parameter number
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters: %d" % total_params)
    print("Total number of trainable parameters: %d" % trainable_params)

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer,
                                                   opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    evaluator = Evaluator(opt, model)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=detection_collate
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train/{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val/{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        if opt.eval_intervals > 0 and epoch % opt.eval_intervals == 0:
            with torch.no_grad():
                metrics, cls_eval = evaluator.eval()
            for k, v in cls_eval.items():
                logger.write("{}: {}\n".format(k, v))
                logger.scalar_summary("val_metrics/zcls_{}".format(k), v, epoch)
            for k, v in metrics.items():
                logger.write("{}: {}\n".format(k, v))
            logger.scalar_summary("val_metrics/mAP", metrics['AP IoU=0.50:0.95 area=all maxDets=100'], epoch)
            logger.scalar_summary("val_metrics/mAP0.50", metrics['AP IoU=0.50      area=all maxDets=100'], epoch)
            logger.scalar_summary("val_metrics/mAR", metrics['AR IoU=0.50:0.95 area=all maxDets=100'], epoch)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts()
    opt = opt.parse()
    main(opt)

cd src
# train freeze
python main.py ctdet --exp_id coco_shelf --arch shelfdet --batch_size 114 --master_batch 18 --lr 5e-4 --val_intervals 1 --eval_intervals 1 --gpus 4,5,6,7 --num_workers 16 --freeze backbone --save_all --print_iter 10
# test
python test.py ctdet --exp_id coco_shelf --arch shelfdet --keep_res --load_model ../exp/ctdet/coco_shelf/model_best.pth

# train
python main.py ctdet --exp_id coco_shelf --arch shelfdet --batch_size 92 --master_batch 20 --lr 3.7e-4 --val_intervals 1 --eval_intervals 1 --gpus 4,5,6,7 --num_workers 16 --save_all --print_iter 10 --load_model ../exp/ctdet/coco_shelf/model_best.pth
# test
python test.py ctdet --exp_id coco_shelf --arch shelfdet --keep_res --load_model ../exp/ctdet/coco_shelf/model_best.pth
cd ..

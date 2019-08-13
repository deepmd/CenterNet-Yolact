cd src
# train freeze
python main.py ctdet --exp_id coco_pelee --arch peleedet --batch_size 114 --master_batch 18 --lr 5e-4 --val_intervals 1 --eval_intervals 1 --gpus 0,1,2,3 --num_workers 16 --freeze backbone --save_all --print_iter 10
# test
python test.py ctdet --exp_id coco_pelee --arch peleedet --keep_res --load_model ../exp/ctdet/coco_pelee/model_best.pth

# train
python main.py ctdet --exp_id coco_pelee --arch peleedet --batch_size 114 --master_batch 26 --lr 5e-4 --val_intervals 1 --eval_intervals 1 --gpus 0,1,2,3 --num_workers 16 --save_all --print_iter 10 --load_model ../exp/ctdet/coco_pelee/model_best.pth
# test
python test.py ctdet --exp_id coco_pelee --arch peleedet --keep_res --load_model ../exp/ctdet/coco_pelee/model_best.pth
cd ..

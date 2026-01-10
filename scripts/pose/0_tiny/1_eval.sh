model_name='lwdetr_tiny_ycbv'
dataset_path='data/datasets/bop/ycbv'
NUM_GPU=1
checkpoint='data/weights/checkpoint0049_new.pth'

python -u -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPU \
                --use_env \
                main.py \
                --eval_only \
                --resume $checkpoint \
                --dataset_file ycbv \
                --eval_set test \
                --rotation_representation '6d' \
                --models /models/  \
                --eval_batches 0 \
                --output_dir output \
                --batch_size 16 \
                --encoder vit_tiny \
                --vit_encoder_num_layers 6 \
                --window_block_indexes 0 2 4 \
                --out_feature_indexes 1 3 5 \
                --dec_layers 3 \
                --group_detr 13 \
                --two_stage \
                --projector_scale P4 \
                --hidden_dim 256 \
                --sa_nheads 8 \
                --ca_nheads 16 \
                --dec_n_points 2 \
                --bbox_reparam \
                --lite_refpoint_refine \
                --num_queries 100 \
                --num_select 100 \
                --dataset_file ycbv \
                --square_resize_div_64 \
                --use_ema \
                --eval \
                --dataset_path $dataset_path \
                --output_dir output/lwdetr_tiny_ycbv

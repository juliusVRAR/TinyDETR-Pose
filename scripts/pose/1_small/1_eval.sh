model_name='lwdetr_small_ycbv'
dataset_path=$1
NUM_GPU=$2
checkpoint=$3

python -u -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPU \
                --use_env \
                main.py \
                --eval_only \
                --resume $checkpoint \
                --dataset_file ycbv \
                --eval_set test \
                --rotation_representation 6d \
                --eval_batches 0 \
                --output_dir output \
                --batch_size 16 \
                --encoder vit_tiny \
                --vit_encoder_num_layers 10 \
                --window_block_indexes 0 1 3 6 7 9 \
                --out_feature_indexes 2 4 5 9 \
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
                --num_queries 50 \
                --num_select 300 \
                --dataset_file ycbv \
                --square_resize_div_64 \
                --use_ema \
                --eval \
                --dataset_path $dataset_path \
                --output_dir output/lwdetr_tiny_ycbv

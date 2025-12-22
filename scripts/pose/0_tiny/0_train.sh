model_name='lwdetr_tiny_ycbv'
dataset_path=$1
NUM_GPU=$2
python -u -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPU \
                --use_env \
                /workspace/LWDETR/main.py \
                            --lr 1e-4 \
                            --lr_transformer 2e-5 \
                            --lr_encoder 1e-5 \
                            --lr_backbone 1e-6 \
                            --batch_size 64 \
                            --weight_decay 1e-4 \
                            --epochs 300 \
                            --lr_drop 60 \
                            --lr_vit_layer_decay 0.8 \
                            --lr_component_decay 0.7 \
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
                            --num_queries 50 \
                            --ia_bce_loss \
                            --cls_loss_coef 1 \
                            --num_select 100 \
                            --dataset_file ycbv \
                            --dataset_path $dataset_path \
                            --square_resize_div_64 \
                            --use_ema \
                            --pretrain_keys_modify_to_load transformer.enc_out_class_embed.0.weight transformer.enc_out_class_embed.1.weight transformer.enc_out_class_embed.2.weight transformer.enc_out_class_embed.3.weight transformer.enc_out_class_embed.4.weight transformer.enc_out_class_embed.5.weight transformer.enc_out_class_embed.6.weight transformer.enc_out_class_embed.7.weight transformer.enc_out_class_embed.8.weight transformer.enc_out_class_embed.9.weight transformer.enc_out_class_embed.10.weight transformer.enc_out_class_embed.11.weight transformer.enc_out_class_embed.12.weight transformer.enc_out_class_embed.0.bias transformer.enc_out_class_embed.1.bias transformer.enc_out_class_embed.2.bias transformer.enc_out_class_embed.3.bias transformer.enc_out_class_embed.4.bias transformer.enc_out_class_embed.5.bias transformer.enc_out_class_embed.6.bias transformer.enc_out_class_embed.7.bias transformer.enc_out_class_embed.8.bias transformer.enc_out_class_embed.9.bias transformer.enc_out_class_embed.10.bias transformer.enc_out_class_embed.11.bias transformer.enc_out_class_embed.12.bias class_embed.weight class_embed.bias \
                            --grayscale \
                            --rgb_augmentation \
                            --tensorboard \
                            --pretrained_encoder /workspace/LWDETR/data/weights/caev2_tiny_S_300e_objects365.pth \
                            --pretrain_weights /workspace/LWDETR/data/weights/LWDETR_tiny_30e_objects365.pth \
                            --matcher_type "6d"

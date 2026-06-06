model_name='lwdetr_tiny_ycbv'
dataset_path=/workspace/LWDETR/data/datasets/bop
NUM_GPU=$1
# Loss balacing coefficients
# Pose loss coefficients
COEF_KPT=$2
COEF_TRANS_XY=$3
COEF_TRANS_Z=$4
COEF_ROT=$5
COEF_ADDS=$6
# Detection loss coefficients
COEF_CLAS=$7
COEF_BBOX=$8
COEF_GIOU=$9
RUN_ID=${10}
JOB_NAME=${11}

ROT_REP=${12:-${ROT_REP:-6d}}
WARM_UP_EPOCHS=${13:-${WARM_UP_EPOCHS:-0}}

# Set this to the checkpoint you want to resume from.
RESUME_CKPT=/workspace/LWDETR/output/pose/0_tiny/80644/_train_tiny_sarr_z15.0_r1.5_adds7.5_wu1_obj365_21:39:46-2026-05-29/checkpoint0009_new.pth
if [ -z "$RUN_ID" ]; then
  RUN_ID=no_slurm_id
fi

if [ ! -f "$RESUME_CKPT" ]; then
  echo "Resume checkpoint not found: $RESUME_CKPT" >&2
  exit 1
fi

# Resume into the same run directory as the checkpoint.
OUTPUT_DIR=$(dirname "$RESUME_CKPT")

python -u -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPU \
                --use_env \
                /workspace/LWDETR/main.py \
                            --lr 1e-4 \
                            --lr_transformer 2e-5 \
                            --lr_encoder 1.5e-4 \
                            --lr_backbone 1e-6 \
                            --weight_decay 1e-4 \
                            --lr_drop 55 \
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
                            --ia_bce_loss \
                            --cls_loss_coef 1 \
                            --square_resize_div_64 \
                            --use_ema \
                            --pretrain_keys_modify_to_load transformer.enc_out_class_embed.0.weight transformer.enc_out_class_embed.1.weight transformer.enc_out_class_embed.2.weight transformer.enc_out_class_embed.3.weight transformer.enc_out_class_embed.4.weight transformer.enc_out_class_embed.5.weight transformer.enc_out_class_embed.6.weight transformer.enc_out_class_embed.7.weight transformer.enc_out_class_embed.8.weight transformer.enc_out_class_embed.9.weight transformer.enc_out_class_embed.10.weight transformer.enc_out_class_embed.11.weight transformer.enc_out_class_embed.12.weight transformer.enc_out_class_embed.0.bias transformer.enc_out_class_embed.1.bias transformer.enc_out_class_embed.2.bias transformer.enc_out_class_embed.3.bias transformer.enc_out_class_embed.4.bias transformer.enc_out_class_embed.5.bias transformer.enc_out_class_embed.6.bias transformer.enc_out_class_embed.7.bias transformer.enc_out_class_embed.8.bias transformer.enc_out_class_embed.9.bias transformer.enc_out_class_embed.10.bias transformer.enc_out_class_embed.11.bias transformer.enc_out_class_embed.12.bias class_embed.weight class_embed.bias \
                            --grayscale \
                            --rgb_augmentation \
                            --tensorboard \
                            --dataset_file ycbv \
                            --dataset_path $dataset_path \
                            --pretrained_encoder /workspace/LWDETR/data/weights/caev2_tiny_S_300e_objects365.pth \
                            --pretrain_weights /workspace/LWDETR/data/weights/LWDETR_tiny_30e_objects365.pth \
                            --epochs 80 \
                            --num_select 100 \
                            --num_queries 50 \
                            --matcher_type "6d" \
                            --batch_size 16 \
                            --n_mesh_points 512 \
                            --keypoint_loss_coef $COEF_KPT \
                            --trans_z_loss_coef $COEF_TRANS_Z \
                            --trans_xy_loss_coef $COEF_TRANS_XY \
                            --rot_loss_coef $COEF_ROT \
                            --adds_loss_coef $COEF_ADDS \
                            --cls_loss_coef $COEF_CLAS \
                            --bbox_loss_coef $COEF_BBOX \
                            --giou_loss_coef $COEF_GIOU \
                            --output_dir $OUTPUT_DIR \
                            --resume $RESUME_CKPT \
                            --warm_up_epochs $WARM_UP_EPOCHS \
                            --rotation_representation $ROT_REP \
                            --reduce_det_loss_epochs 50000000 \
                            --quick_eval \
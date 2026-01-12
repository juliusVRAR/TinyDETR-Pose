python visualize_pose_predictions.py \
  --checkpoint /workspace/LWDETR/data/weights/checkpoint0046_new.pth \
  --split test \
  --dataset_path /workspace/LWDETR/data/datasets/bop/ycbv \
  --dataset_file ycbv \
  --encoder vit_tiny \
  --vit_encoder_num_layers 6 \
  --window_block_indexes 0 2 4 \
  --out_feature_indexes 1 3 5 \
  --n_mesh_points 1024 \
  --dec_layers 3 --group_detr 13 --two_stage \
  --projector_scale P4 --hidden_dim 256 \
  --sa_nheads 8 --ca_nheads 16 --dec_n_points 2 \
  --bbox_reparam --lite_refpoint_refine --num_queries 100 \
  --ia_bce_loss --cls_loss_coef 1 --num_select 100 \
  --square_resize_div_64 --use_ema \
  --pretrained_encoder /workspace/LWDETR/data/weights/caev2_tiny_S_300e_objects365.pth \
  --matcher_type 6d --keypoint_loss_coef 10.0 \
  --trans_z_loss_coef 1.0 --trans_xy_loss_coef 1.0 \
  --rot_loss_coef 1.0 --adds_loss_coef 1.0 \
  --num_images 16 --score_threshold 0.4 --vis_output output/vis_test

  # --pretrain_weights /workspace/LWDETR/data/weights/LWDETR_xlarge_30e_objects365.pth \
